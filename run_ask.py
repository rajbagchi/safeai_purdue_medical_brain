r"""
One-shot Q&A against pipeline v2 (loads or builds KB, prints VHT answer).

  pip install -r requirements.txt
  $env:SAFEAI_USE_LOCAL_LLM="1"
  $env:SAFEAI_LLM_GGUF="C:\models\qwen2.5-3b-instruct-q4_k_m.gguf"
  python run_ask.py --preset uganda --query "When should I refer a child with fever?"

Optional: --rebuild-kb  (delete KB json before init)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_DEFAULT_GGUF = r"C:\models\qwen2.5-3b-instruct-q4_k_m.gguf"


def _preset(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "-")
    m = {
        "who-malaria": "who-malaria",
        "who-malaria-nih": "who-malaria",
        "malaria": "who-malaria",
        "uganda": "uganda",
        "uganda-clinical-2023": "uganda",
        "uganda_clinical": "uganda",
    }
    if key not in m:
        raise SystemExit(f"Unknown preset {name!r}. Use: {sorted(set(m.keys()))}")
    return m[key]


def main() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Pipeline v2 — single ask")
    p.add_argument("--preset", default="uganda")
    p.add_argument("--query", required=True)
    p.add_argument("--pdf", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--rebuild-kb", action="store_true")
    p.add_argument("--gguf", default=os.environ.get("SAFEAI_LLM_GGUF") or _DEFAULT_GGUF)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    preset = _preset(args.preset)
    gguf = (args.gguf or "").strip()
    if gguf and Path(gguf).is_file():
        os.environ["SAFEAI_LLM_GGUF"] = gguf
        os.environ["SAFEAI_USE_LOCAL_LLM"] = "1"

    from pipeline.compat import fix_stdio_encoding
    from pipeline.config import (
        extraction_config_uganda_clinical_2023,
        extraction_config_who_malaria_nih,
    )
    from pipeline.orchestrator import MedicalQASystem

    fix_stdio_encoding()

    kw: dict = {}
    if args.pdf:
        kw["pdf_path"] = args.pdf
    if args.output_dir:
        kw["output_dir"] = args.output_dir
    if preset == "who-malaria":
        cfg = extraction_config_who_malaria_nih(**kw) if kw else extraction_config_who_malaria_nih()
    else:
        cfg = extraction_config_uganda_clinical_2023(**kw) if kw else extraction_config_uganda_clinical_2023()

    if not Path(cfg.pdf_path).is_file():
        raise SystemExit(f"PDF not found: {cfg.pdf_path}")

    kb = Path(cfg.output_dir) / "knowledge_base.json"
    if args.rebuild_kb and kb.is_file():
        kb.unlink()

    qa = MedicalQASystem(config=cfg)
    qa.initialize()

    result = qa.answer_with_response(
        args.query,
        use_local_llm=True if Path(gguf).is_file() else None,
        local_llm_gguf=gguf if Path(gguf).is_file() else None,
        allow_client_supplied_gguf_path=True,
    )

    structured = result.get("structured")
    if structured is not None:
        result = dict(result)
        result["structured"] = asdict(structured)
        if "triage" in result["structured"] and hasattr(structured.triage, "name"):
            result["structured"]["triage"] = structured.triage.name
    tri = result.get("triage")
    if hasattr(tri, "name"):
        result = dict(result)
        result["triage"] = tri.name

    print(result.get("vht_response", ""))
    print("\n---\n")
    print(
        json.dumps(
            {
                "local_llm_used": result.get("local_llm_used"),
                "vht_synthesis_mode": result.get("vht_synthesis_mode"),
                "vht_retrieval_pool_k": result.get("vht_retrieval_pool_k"),
                "local_llm_skipped_reason": result.get("local_llm_skipped_reason"),
            },
            indent=2,
        )
    )

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
