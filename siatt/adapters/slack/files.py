"""Fetching what came attached, without handing out the bot token.

This is the one place in Siatt that sends an `Authorization` header to an
address it did not choose. `web_fetch` deliberately never does
(`siatt/fetch/client.py`), and the difference is not squeamishness: a private
Slack file is only readable with the bot token, and the URL that names it
arrives *inside an event payload*. Following that URL wherever it points, with
the token attached, is an SSRF that gives the token away.

So three rules, all of them before the request is built:

1. **https, and a host on the allowlist.** Not "resolve it and judge the
   address", which is what `siatt/fetch/guard.py` does for a URL a model chose:
   here we know exactly whose file store this should be, and anything else is
   already wrong. A name check is the stronger check when there is only one
   right answer.
2. **No redirects.** An authorized `url_private_download` answers 200. A 302 is
   Slack saying "not you" and pointing at a login page, and the only sane thing
   to do with it is to stop -- following it would put the token on the next hop.
3. **The cap is on the bytes, not on the claim.** `size` in the payload is
   somebody else's number. The stream is abandoned when the real bytes pass the
   limit.

Then, after the bytes: what arrived has to look like what was promised. The
well-known way to get this feature wrong is to store Slack's HTML login page as
a JPEG, because the install is missing `files:read` and nobody checked.

None of it runs inside the ack. This is an `EventPreparer` -- after the queue,
before the turn -- so a 20MB download never sits in front of the three seconds
Slack allows, and a redelivered inbox row re-fetches into a content-addressed
store where the second write is a no-op.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from siatt.core.events import Attached, InboundEvent
from siatt.store.blobs import AttachmentError, Attachments

log = logging.getLogger(__name__)

#: Where Slack serves uploaded files from. Suffixes: an exact host, or anything
#: under it. `files.slack.com` is what every install sees; the rest of
#: `slack.com` is here because an Enterprise Grid workspace is served from its
#: own subdomain, and refusing those would break the feature for exactly the
#: installs most likely to care about it.
#:
#: Note what a *suffix* means here — `_permitted` matches on a dotted boundary,
#: so `slack.com.evil.example` is not under `slack.com`. Written as a check
#: rather than as `endswith`, because `endswith` is how this goes wrong.
DEFAULT_FILE_HOSTS: tuple[str, ...] = ("slack.com",)

#: Off the wire, per read. Small enough that the cap stops an oversized file
#: early rather than after it is all in memory.
CHUNK = 64 * 1024

#: The whole download. A file is a detour on a turn somebody is waiting through,
#: and this sits behind the queue rather than in front of the ack, so it can be
#: generous without costing anybody an acknowledgement.
DEFAULT_TIMEOUT = 60.0

#: What an HTML document starts with, lowercased. The login page Slack serves to
#: an unauthorized request is the one wrong body that looks entirely successful:
#: 200, a sensible length, and nothing about it says it is not a photograph
#: except the bytes themselves.
_HTML_MARKERS = (b"<!doctype", b"<html", b"<head", b"<?xml")

#: Magic numbers, as (offset, bytes, family). Not a full sniffer -- there is no
#: attempt to name the codec, only to answer "is this the *kind* of thing it
#: said it was". An unrecognized file is accepted on the surface's word, because
#: an allowlist of container signatures would silently drop whatever format
#: phones start writing next.
_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image"),
    (0, b"\xff\xd8\xff", "image"),
    (0, b"GIF87a", "image"),
    (0, b"GIF89a", "image"),
    (0, b"BM", "image"),
    (0, b"II*\x00", "image"),
    (0, b"MM\x00*", "image"),
    (0, b"\x1a\x45\xdf\xa3", "video"),  # Matroska / WebM
    (0, b"OggS", "video"),
    (0, b"FLV\x01", "video"),
)

#: Containers that are told apart by a brand rather than by the first bytes.
#: RIFF is WebP (an image) and AVI (a video); ISO base media is HEIC and AVIF
#: (images) and MP4 and QuickTime (videos).
_RIFF_BRANDS = {b"WEBP": "image", b"AVI ": "video"}
_ISOBMFF_BRANDS = {
    b"heic": "image",
    b"heix": "image",
    b"heim": "image",
    b"heis": "image",
    b"mif1": "image",
    b"msf1": "image",
    b"avif": "image",
    b"avis": "image",
}


class SlackFileError(AttachmentError):
    """One attachment could not be fetched. Never fails the turn."""


@dataclass(frozen=True, slots=True)
class Stored:
    """What became of one attachment, as a fact rather than as a plan."""

    name: str
    #: None when the bytes were not kept, whatever the reason.
    sha256: str | None = None
    #: Why not, in words a person reading the transcript would understand. The
    #: model sees this: it is what turns "I cannot read that" into "I cannot
    #: read that *because*", which is the difference between a dead end and
    #: something the person can fix.
    reason: str | None = None


def note(stored: list[Stored]) -> str:
    """The block appended to a message that came with files.

    Composed here rather than at ingress because ingress cannot know any of it.
    Without a note the agent sees "what's in this?" with nothing in it, and no
    way to tell that a file is what it is being asked about -- naming them is
    what lets it say what it can and cannot do with them.
    """
    if not stored:
        return ""
    lines = [
        f"- {_shown(item.name)} — "
        + ("stored, but Siatt cannot read it yet" if item.sha256 else f"not stored: {item.reason}")
        for item in stored
    ]
    head = "[attached]" if len(stored) == 1 else f"[attached: {len(stored)} files]"
    return "\n".join([head, *lines])


#: A filename, shown. Whoever uploaded the file chose it, and it lands in a
#: block the model reads as Siatt's own annotation rather than as somebody's
#: message — so a name holding a newline could forge a line of that block, and
#: claim about a second file whatever it liked. Collapsing the whitespace is
#: what makes one file one line.
_SHOWN_CHARS = 120


def _shown(name: str) -> str:
    flat = " ".join(name.split())
    if len(flat) > _SHOWN_CHARS:
        return f"{flat[:_SHOWN_CHARS]}…"
    return flat or "an untitled file"


def with_note(text: str, stored: list[Stored]) -> str:
    body = note(stored)
    if not body:
        return text
    return f"{text}\n\n{body}" if text else body


class SlackFiles:
    """Downloads Slack's uploads into the attachment store, or explains why not.

    One per adapter. Holds a connection pool and the bot token, and hands out
    neither: `collect` is the whole surface.
    """

    def __init__(
        self,
        attachments: Attachments | None,
        *,
        token: str,
        hosts: tuple[str, ...] = DEFAULT_FILE_HOSTS,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._attachments = attachments
        self._token = token
        self._hosts = hosts
        # follow_redirects is off and must stay off. See rule 2 above: a
        # redirect on this path is an authorization failure wearing a 302, and
        # following it would put the bot token on a host nothing has judged.
        self._client = client or httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, headers={"User-Agent": "SiattBot/1.0"}
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def collect(self, event: InboundEvent) -> InboundEvent:
        """Fetch what came with `event`, and say in its text what happened.

        Never raises. This sits between the queue and the agent, exactly where
        `Directory.hydrate` sits, and for the same reason: a file store that is
        down is a reason to answer without the file, not a reason to fail the
        turn and have the message redelivered until its retry budget runs out.
        """
        if not event.attachments:
            return event
        try:
            stored = [await self._one(item, event) for item in event.attachments]
        except Exception:
            log.exception("could not fetch Slack attachments for %s", event.external_id)
            return event
        return event.model_copy(update={"text": with_note(event.text, stored)})

    async def _one(self, item: Attached, event: InboundEvent) -> Stored:
        name = item.name or "an untitled file"
        if self._attachments is None:
            return Stored(name, reason="this install does not keep attachments")
        if item.url is None:
            return Stored(name, reason="Slack gave no address for it")
        if item.mime and not self._attachments.accepts(item.mime):
            return Stored(name, reason=f"{item.mime} is not a kind Siatt keeps")
        try:
            host = _permitted(item.url, self._hosts)
        except SlackFileError as exc:
            # Loud, because this one is not an accident. A `url_private` that is
            # not on Slack is an event payload trying to be sent somewhere with
            # a token attached, and the interesting fact is that it happened.
            log.warning("refused a Slack attachment URL on %s: %s", event.external_id, exc)
            return Stored(name, reason=str(exc))
        try:
            sha256 = await self._download(item, host=host, event=event)
        except SlackFileError as exc:
            log.info("could not fetch %s on %s: %s", name, event.external_id, exc)
            return Stored(name, reason=str(exc))
        return Stored(name, sha256=sha256)

    async def _download(self, item: Attached, *, host: str, event: InboundEvent) -> str:
        assert item.url is not None and self._attachments is not None
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            async with self._client.stream("GET", item.url, headers=headers) as response:
                _judge(response, host=host)
                return await self._attachments.put(
                    _verified(response, item),
                    mime=item.mime or "application/octet-stream",
                    source_name="slack",
                    scope=event.scope,
                    external_id=event.external_id,
                    session_id=event.session_id,
                    author=event.author,
                    name=item.name,
                )
        except httpx.HTTPError as exc:
            raise SlackFileError(f"Slack did not serve it ({type(exc).__name__})") from exc
        except AttachmentError as exc:
            raise SlackFileError(str(exc)) from exc


def _permitted(url: str, hosts: tuple[str, ...]) -> str:
    """The host, or a refusal. Nothing is sent before this returns.

    The URL came out of an event payload, and the request that follows carries
    the bot token. Every branch here is a way that request could have gone to
    somebody else.
    """
    parts = urlsplit(url.strip())
    if parts.scheme.lower() != "https":
        # Never http. A bearer token on a cleartext hop is given away to
        # everything between here and there, and Slack has no such endpoint —
        # an http URL in this field is already somebody else's idea.
        raise SlackFileError("only https attachment URLs are fetched")
    if parts.username or parts.password:
        raise SlackFileError("an attachment URL with credentials in it will not be fetched")
    host = (parts.hostname or "").lower()
    if not host:
        raise SlackFileError("that attachment URL has no host in it")
    if parts.port not in (None, 443):
        raise SlackFileError(f"only port 443 is fetched, and that URL asks for {parts.port}")
    if not any(host == allowed or host.endswith(f".{allowed}") for allowed in hosts):
        # The dotted boundary is the point. A bare `endswith` would accept
        # `slack.com.evil.example`, which is the whole attack.
        raise SlackFileError(f"{host} is not one of Slack's file hosts")
    return host


def _judge(response: httpx.Response, *, host: str) -> None:
    """What the response line and headers already say, before any bytes."""
    if response.is_redirect:
        # The signature of a missing scope. Slack answers an unauthorized
        # private-file request by redirecting to a sign-in page, so this is
        # almost never a moved file.
        raise SlackFileError(
            f"{host} redirected instead of serving the file — Siatt's Slack app "
            "probably lacks the files:read scope"
        )
    if response.status_code != 200:
        raise SlackFileError(f"{host} answered {response.status_code}")


async def _verified(response: httpx.Response, item: Attached) -> AsyncIterator[bytes]:
    """The body, with the first chunk checked before any of it is kept.

    Wrapping the stream rather than buffering it: the check needs the head of
    the file and the store needs all of it, and reading it twice would mean
    holding a video in memory to look at its first eight bytes.
    """
    head = b""
    async for chunk in response.aiter_bytes(CHUNK):
        if not head:
            head = chunk
            _check(head, item)
        yield chunk


def _check(head: bytes, item: Attached) -> None:
    """Whether what arrived is the kind of thing that was promised."""
    if _looks_like_markup(head):
        raise SlackFileError(
            "Slack served a web page rather than the file — Siatt's Slack app "
            "probably lacks the files:read scope"
        )
    family = _family(head)
    claimed = item.mime.split("/", 1)[0].lower()
    if family is not None and claimed and family != claimed:
        # Only when both are known. An unrecognized container is taken on the
        # surface's word; a JPEG announced as a video is not.
        raise SlackFileError(f"the bytes are {family}, not the {item.mime} Slack described")


def _looks_like_markup(head: bytes) -> bool:
    start = head[:512].lstrip().lower()
    return any(start.startswith(marker) for marker in _HTML_MARKERS)


def _family(head: bytes) -> str | None:
    """`image`, `video`, or None when nothing here is recognized."""
    for offset, magic, family in _SIGNATURES:
        if head[offset : offset + len(magic)] == magic:
            return family
    if head[:4] == b"RIFF":
        return _RIFF_BRANDS.get(head[8:12])
    if head[4:8] == b"ftyp":
        # ISO base media: the brand says whether it is a still or a movie, and
        # everything unlisted (mp42, isom, qt) is a movie.
        return _ISOBMFF_BRANDS.get(head[8:12], "video")
    return None
