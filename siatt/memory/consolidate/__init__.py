"""Safe boundary between consolidation models and long-term memory."""

from siatt.memory.consolidate.prompt import (
    ConsolidationInput,
    build_request,
    decode_plan,
    untrusted_block,
)

__all__ = ["ConsolidationInput", "build_request", "decode_plan", "untrusted_block"]
