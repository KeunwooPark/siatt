"""Token counting for the context packer.

The packer needs an estimate that is *conservative* — over-counting costs a
little unused context, under-counting costs a `ContextOverflowError` mid-turn.
Every heuristic here therefore rounds against us.
"""

from __future__ import annotations

from typing import Protocol

from siatt.llm.types import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

#: Rough per-message framing cost (role markers, delimiters). Both provider
#: families add something in this range.
MESSAGE_OVERHEAD_TOKENS = 4

#: Tool definitions are serialized into the prompt; their JSON schema costs
#: roughly this much per character.
_ASCII_TOKENS_PER_CHAR = 0.25

#: What one image can cost at most. Both provider families resize anything
#: larger down to roughly 1568px on the long edge before charging for it, so
#: this is a real ceiling rather than a guess -- which is what makes it safe to
#: charge it for an image whose header we could not read.
MAX_IMAGE_TOKENS = 1_600

#: The divisor both families document: an image costs about its area in pixels
#: over this.
PIXELS_PER_TOKEN = 750

#: Non-ASCII text (CJK especially) tokenizes far denser than Latin script.
#: One token per character over-estimates for accented Latin and under-estimates
#: for nothing, which is the direction we want.
_WIDE_TOKENS_PER_CHAR = 1.0


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """Dependency-free estimator. The default.

    Accurate to roughly ±20% on English prose and code, and deliberately
    pessimistic on non-Latin scripts.
    """

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        ascii_chars = sum(1 for ch in text if ch.isascii())
        wide_chars = len(text) - ascii_chars
        estimate = ascii_chars * _ASCII_TOKENS_PER_CHAR + wide_chars * _WIDE_TOKENS_PER_CHAR
        return max(1, int(estimate + 0.5))


class TiktokenTokenizer:
    """Exact counts for OpenAI models, when the optional extra is installed."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        import tiktoken  # imported lazily; the dependency is optional

        self.name = f"tiktoken:{encoding_name}"
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


def default_tokenizer() -> Tokenizer:
    """Best available tokenizer.

    Falls back silently: an exact count is nice, but the packer is designed to
    be correct with an estimate, so a missing optional dependency must never be
    a startup failure.
    """
    try:
        return TiktokenTokenizer()
    except Exception:
        return HeuristicTokenizer()


def image_tokens(width: int | None, height: int | None) -> int:
    """What one image costs, rounded against us.

    Unknown dimensions cost the ceiling. This module's rule is that
    over-counting wastes a little context and under-counting ends the turn, and
    an image is the block where that difference is three orders of magnitude.
    """
    if width is None or height is None or width <= 0 or height <= 0:
        return MAX_IMAGE_TOKENS
    return min(MAX_IMAGE_TOKENS, max(1, -(-width * height // PIXELS_PER_TOKEN)))


def count_message(msg: Message, tokenizer: Tokenizer) -> int:
    total = MESSAGE_OVERHEAD_TOKENS
    for block in msg.content:
        match block:
            case TextBlock():
                total += tokenizer.count(block.text)
            case ThinkingBlock():
                total += tokenizer.count(block.thinking)
            case ImageBlock():
                # By area, not by the length of its JSON. A picture serializes
                # to a hash and a mime type — about thirty tokens — and costs
                # up to sixteen hundred, so counting it as text is how a
                # context that looked like it fit arrives as a 400.
                total += image_tokens(block.width, block.height)
            case ToolUseBlock():
                total += tokenizer.count(block.name) + tokenizer.count(repr(block.input))
            case ToolResultBlock():
                total += tokenizer.count(block.content)
    return total


def count_messages(messages: tuple[Message, ...] | list[Message], tokenizer: Tokenizer) -> int:
    return sum(count_message(m, tokenizer) for m in messages)
