"""Setting up a standing task by asking for one.

#179 gave Siatt a table of standing tasks and a CLI to manage it. That is the
wrong surface for the person it is for: somebody in a Slack thread who says
"do this every weekday morning" should not be told to open a terminal and
write five cron fields.

The translation is the one part of this a model is genuinely good at — "every
weekday at 9am Seoul time" into `0 9 * * 1-5` and `Asia/Seoul` — so the model
does it and the tool validates rather than trusts. `Cron.parse` with the zone,
the interval floor, the per-owner cap: every refusal comes back as a tool error
the model can read and correct, which is what `ToolRegistry` already gives
these for free.

Two constraints shape all three, and they are the same ones the memory tools
work under.

**The session supplies the destination, never the model.** `session_id`,
`channel`, `reply_to`, `scope` and `owner` come off `ToolContext`, which the
turn built from the event. There is no argument for where a task posts, so no
instruction smuggled into text Siatt read can create one that posts anywhere but
the conversation it was created in. A task created in a DM stays in the DM
(§11.1).

`new_thread` is not an exception to that, and the distinction is worth being
exact about because it is the one an injected instruction would try to blur. It
chooses the *shape* of the answer — its own thread in this channel, rather than
another message in the thread this was asked in — and not the place: it resolves
to `channel`, off the same `ToolContext`, which is the channel these words were
already said in. The set of places a tool can reach is still exactly one, and it
is still the caller's own. Pointing a schedule at a *different* channel is
operator work, `siatt task add --destination` against a name in
`[tasks.destinations]`, and there is deliberately no way to ask for it here.

**Listing and cancelling are scoped to the calling channel**, and to the asker,
for the same reason: text arriving in one channel must not be able to enumerate
or delete another channel's schedules (§7.1). The narrowing is in the query,
not applied to the results — a tool that read every row and then dropped the
ones it should not show has already had them.

The channel and not the thread, because the thread was the wrong unit and made
a live schedule read as a deleted one (#260). A task's `session_id` is fixed to
the thread it was created in, so from any other thread in the channel it posts
to, `schedule_list` answered "there are no standing tasks in this conversation"
— true, and taken for "the schedule is gone". The reply to that is a duplicate
9am digest.

Widening to the channel widens nothing anyone can reach: it is the asker's own
task, in the channel these words were already said in, and it is the same one
place `schedule_create` can reach. See `_visible`, which both tools read, so
that what can be listed is exactly what can be cancelled.

What `schedule_create` returns is the next three fire times, rendered in the
task's own zone. That is the whole confirmation story: nobody can check
`0 9 * * 1-5`, and anybody can check "Mon 08 Sep 09:00 Asia/Seoul".
"""

from __future__ import annotations

import logging
from typing import Any

from siatt.config import HERE
from siatt.core.tools import Tool, ToolContext
from siatt.runner.cron import CronError
from siatt.runner.tasks import ACTIVE, Task, TaskError, Tasks, render_fires

log = logging.getLogger(__name__)

#: How many fire times to read back. Three is enough to show the shape of a
#: weekly or a weekday schedule — one is not: "Mon 08 Sep 09:00" alone is
#: equally consistent with every day, every weekday and every Monday.
CONFIRM_FIRES = 3

#: A task with no owner is one nothing can cap, cancel or notify. Only reachable
#: from a surface that has no notion of who is speaking, which today is the
#: terminal — and the terminal has `siatt task add`.
NO_OWNER = (
    "This conversation has no user identity, so it cannot own a schedule. "
    "Standing tasks are set up from a chat surface, or with `siatt task add`."
)


def schedule_tools(tasks: Tasks) -> list[Tool]:
    """The three scheduling tools, bound to one tasks table."""
    return [_create_tool(tasks), _list_tool(tasks), _cancel_tool(tasks)]


# -- schedule_create ---------------------------------------------------------


def _create_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        try:
            task = await tasks.create(
                owner=context.author,
                # Every one of these is the session's. None is an argument.
                surface=_surface_of(context.session_id),
                session_id=context.session_id,
                channel=context.channel,
                reply_to=context.reply_to,
                scope=context.scope,
                prompt=str(args["prompt"]),
                cron=str(args["cron"]),
                timezone=_timezone(args),
                fire_once=bool(args.get("fire_once", False)),
                # Still not a destination: `here` *is* `context.channel`, which
                # is where this task was already going to post.
                destination=HERE if args.get("new_thread") else None,
            )
        except TaskError as exc:
            # A refusal, not a crash: the model wrote the expression, and it is
            # what can correct it. The floor and the cap both name their number
            # in the message for exactly that reason.
            return f"That schedule was not created. {exc}"
        fires = render_fires(task.next_fires(CONFIRM_FIRES), task.timezone)
        listed = "\n".join(f"- {fire}" for fire in fires)
        return f"Created schedule {task.id}. It next runs:\n{listed}"

    return Tool(
        name="schedule_create",
        description=(
            "Set up a standing task: something to do again on a schedule, in "
            "this conversation. Translate what the person asked for into a "
            "five-field cron expression and an IANA time zone — 'every weekday "
            "at 9am Seoul time' is `0 9 * * 1-5` with `Asia/Seoul`. Use the "
            "zone the person named, or the one you already know they are in; "
            "omit it only when they meant UTC. The prompt is what you will be "
            "asked each time it fires, so write it as an instruction that will "
            "still make sense with none of this conversation around it. "
            "Returns the next few fire times: state them back before treating "
            "the task as set up, because that is the only part of this the "
            "person can actually check."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to do each time, written to stand on its own — "
                        "'search for what happened in AI overnight and give me "
                        "the five that matter', not 'do that again'."
                    ),
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "Five fields: minute hour day-of-month month day-of-week. "
                        "`0 9 * * 1-5` is every weekday at nine."
                    ),
                },
                "timezone": {
                    "type": "string",
                    "description": (
                        "IANA zone the hour is read in, e.g. 'Asia/Seoul'. Not an "
                        "abbreviation: 'KST' is not a zone. Omit for UTC."
                    ),
                },
                "fire_once": {
                    "type": "boolean",
                    "description": "Run it once and finish, rather than every time it matches.",
                },
                "new_thread": {
                    "type": "boolean",
                    "description": (
                        "Post each run as a new thread in this channel, instead of "
                        "replying in this one. Use it for something recurring that "
                        "people will want to discuss on its own — a morning "
                        "briefing — and leave it off for a reminder that belongs "
                        "with the conversation it came out of. Either way it posts "
                        "here; there is no way to send a schedule somewhere else."
                    ),
                },
            },
            "required": ["prompt", "cron"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _timezone(args: dict[str, Any]) -> str | None:
    zone = str(args.get("timezone") or "").strip()
    return zone or None


def _surface_of(session_id: str) -> str:
    """Which surface a session key belongs to.

    Session ids are built by the adapter as `<surface>:...`, so the prefix is
    the surface. A task's own `surface` is what its fire is delivered as, and
    getting it wrong would enqueue an event no adapter answers.
    """
    surface, _, _ = session_id.partition(":")
    return surface or "cli"


# -- what the caller may see -------------------------------------------------


async def _visible(tasks: Tasks, context: ToolContext) -> list[Task]:
    """The asker's own schedules that answer where they are asking (#260).

    *Where* is the channel when there is one, and the session when there is
    not. A task's `session_id` is the thread it was created in and is fixed
    there, so narrowing on it made a schedule invisible from every other thread
    in the channel it posts to — including, in the failure this comes from, the
    thread somebody went to to ask about it. `schedule_list` answered "there
    are no standing tasks in this conversation", which was true, and was read
    as "the schedule is gone", which was not: two 9am digests were alive and
    had run that morning. The model offered a replacement, and only because the
    person knew better was a third 9am digest not left behind.

    Widening from the thread to the channel is not a widening of what anybody
    can reach. `owner` still bounds it to the asker's own, and a channel is
    where these words were already said and where the answer is about to be
    posted — the same one place `schedule_create` can reach, and the same
    argument §11.1 makes about it. A DM's channel is the DM, so a DM lists the
    DM's. What text arriving in one channel still cannot do is enumerate
    another channel's schedules, which is what §7.1 is about.

    One helper because `schedule_cancel` reads it too, and must: it finds a
    task under this narrowing rather than by id, so that an id from somewhere
    else comes back as "no such schedule" rather than as a refusal confirming
    one exists. A listing that offered ids the next tool refused would be worse
    than either scoping on its own.
    """
    if not context.author:
        # Both callers check this first and answer `NO_OWNER`. Checked again
        # here because the failure if one ever stops is silent and total:
        # `owner=None` is not "nobody's", it is *no narrowing at all*, and this
        # would hand back every schedule in the channel.
        return []
    if context.channel:
        return await tasks.all(owner=context.author, channel=context.channel)
    return await tasks.all(owner=context.author, session_id=context.session_id)


def _here(context: ToolContext) -> str:
    """What "here" meant, for a listing that found nothing.

    "None" has to keep meaning none. A tool that narrowed and then reported an
    empty result without saying what it narrowed to is how #260 happened, and
    saying it costs six words.
    """
    return "this channel" if context.channel else "this conversation"


# -- schedule_list -----------------------------------------------------------


def _list_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        found = await _visible(tasks, context)
        if not found:
            return f"You have no standing tasks in {_here(context)}."
        return "\n".join(_describe(task, context) for task in found)

    return Tool(
        name="schedule_list",
        description=(
            "List the standing tasks you set up in this channel, with their "
            "schedules, where each one posts, and when each next runs. Use it "
            "before cancelling one, and to answer 'what have you got "
            "scheduled?'. It covers the whole channel, not only this thread — "
            "a schedule set up in one thread posts to the channel and is "
            "listed from any of them. It cannot see another channel's."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
    )


def _describe(task: Task, context: ToolContext) -> str:
    return (
        f"{task.id} — {task.prompt!r} ({task.label}), {task.state}, "
        f"{_posts(task, context)}, next: {_next(task)}"
    )


def _posts(task: Task, context: ToolContext) -> str:
    """Where this one answers, so two 9am digests are told apart.

    The whole point of listing a channel rather than a thread: the reader now
    sees schedules they did not set up *here*, and "posts every morning at 9"
    twice over with nothing to distinguish them is a listing that has replaced
    one confusion with another.
    """
    if task.destination == HERE:
        return "posts as a new thread in this channel"
    if task.destination is not None:
        return f"posts to {task.destination!r}"
    if task.session_id == context.session_id:
        return "posts in this thread"
    return "posts in the thread it was created in"


def _next(task: Task) -> str:
    """When it fires next, or why it never will.

    A schedule that stopped parsing has to say so here. Answering "what have
    you got scheduled?" with a row and no next time reads as though it is
    fine, and the person would go on believing they get their nine o'clock.
    """
    if task.state != ACTIVE:
        return f"not while it is {task.state}"
    try:
        return render_fires(task.next_fires(1), task.timezone)[0]
    except CronError as exc:
        return f"never — {exc}"


# -- schedule_cancel ---------------------------------------------------------


def _cancel_tool(tasks: Tasks) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        if not context.author:
            return NO_OWNER
        task_id = str(args["id"]).strip()
        # Found under the caller's own narrowing rather than by id and then
        # checked: an id from another channel must come back as "no such
        # schedule", not as a refusal that confirms it exists.
        #
        # The same narrowing `schedule_list` reads, and that is the invariant
        # rather than an accident of sharing a helper: you may cancel exactly
        # what you may see. Widening the listing alone would hand the model ids
        # this refuses (#260).
        visible = await _visible(tasks, context)
        if task_id not in {task.id for task in visible}:
            return f"There is no schedule {task_id!r} in {_here(context)}."
        await tasks.cancel(task_id)
        return f"Cancelled schedule {task_id}. It will not run again."

    return Tool(
        name="schedule_cancel",
        description=(
            "Cancel a standing task in this channel, by the id "
            "`schedule_list` gives — exactly what that lists is what this can "
            "cancel. Stopping it is immediate and permanent; there is nothing "
            "to undo it with."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The schedule's id."},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        handler=handler,
    )
