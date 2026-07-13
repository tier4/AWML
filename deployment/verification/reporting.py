"""Shared rendering helpers for verification logs.

One banner width and one pass/fail vocabulary, so the verifier and the verification
orchestrator render results identically instead of each hard-coding their own.
"""

from __future__ import annotations

#: Log-banner width used across the verification path (matches the evaluation orchestrator).
BANNER_WIDTH = 80


def banner(char: str = "=") -> str:
    """Return a full-width banner line."""
    return char * BANNER_WIDTH


def format_verdict(passed: bool) -> str:
    """Render a pass/fail verdict token (single source for the ✓/✗ vocabulary)."""
    return "PASSED ✓" if passed else "FAILED ✗"
