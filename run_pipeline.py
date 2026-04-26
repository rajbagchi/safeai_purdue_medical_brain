"""
Pipeline v2 (capstone2): extraction → validation → chunking → Q&A with
search-first local LLM VHT synthesis.

  python run_pipeline.py
  python -m pipeline
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pipeline.compat import fix_stdio_encoding

fix_stdio_encoding()

from pipeline.cli import main

if __name__ == "__main__":
    main()
