"""Cross-platform helpers (e.g. Windows console encoding)."""

from __future__ import annotations

import sys


def fix_stdio_encoding() -> None:
    """
    Use UTF-8 with replacement for undecodable output on stdout/stderr.

    Pipeline progress messages use emoji; Windows often defaults to cp1252
    (charmap), which raises UnicodeEncodeError during print().
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError, TypeError):
            pass
