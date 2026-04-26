"""
FastAPI service for pipeline v2 (capstone2): BM25 + guardrails + search-first VHT LLM.

  pip install -r requirements-api.txt
  uvicorn api:app --host 0.0.0.0 --port 8001

Default KB output dirs are under this repo (``kb_uganda_clinical_2023``, ``kb_who_malaria``).
Source PDFs remain ``C:\\temp\\capstone\\...`` per ``pipeline.config``.

Local LLM:
  SAFEAI_USE_LOCAL_LLM=1
  SAFEAI_LLM_GGUF=C:\\models\\qwen2.5-3b-instruct-q4_k_m.gguf

Legacy v1-style LLM rewrite (draft + excerpts): set SAFEAI_VHT_LLM_LEGACY=1.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Optional

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.compat import fix_stdio_encoding

fix_stdio_encoding()

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from pipeline.config import (
    ExtractionConfig,
    extraction_config_uganda_clinical_2023,
    extraction_config_who_malaria_nih,
)
from pipeline.orchestrator import MedicalQASystem

app = FastAPI(
    title="SafeAI Clinical Pipeline v2 (search-first VHT)",
    description=(
        "Uganda Clinical Guidelines 2023 & WHO Malaria (NCBI) — BM25 + guardrail + "
        "VHT layer with search-first local LLM synthesis by default"
    ),
    version="2.0.0",
)

_qa: Optional[MedicalQASystem] = None
_loaded: Optional[Dict[str, Any]] = None

_PRESET_ALIASES = {
    "who-malaria": "who-malaria",
    "who-malaria-nih": "who-malaria",
    "malaria": "who-malaria",
    "uganda": "uganda",
    "uganda-clinical-2023": "uganda",
    "uganda_clinical": "uganda",
}


def _normalize_preset(name: str) -> str:
    key = (name or "").strip().lower().replace(" ", "-")
    if key not in _PRESET_ALIASES:
        allowed = sorted(set(_PRESET_ALIASES.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{name}'. Use one of: {allowed}",
        )
    return _PRESET_ALIASES[key]


def _build_config(preset: str, pdf_path: Optional[str], output_dir: Optional[str]) -> ExtractionConfig:
    def _kwargs() -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        if pdf_path:
            kw["pdf_path"] = pdf_path
        if output_dir:
            kw["output_dir"] = output_dir
        return kw

    kw = _kwargs()
    if preset == "who-malaria":
        return extraction_config_who_malaria_nih(**kw) if kw else extraction_config_who_malaria_nih()
    return extraction_config_uganda_clinical_2023(**kw) if kw else extraction_config_uganda_clinical_2023()


def _serialize_structured(obj: Any) -> Dict[str, Any]:
    d = asdict(obj)
    if "triage" in d and hasattr(obj.triage, "name"):
        d["triage"] = obj.triage.name
    return d


def _serialize_ask_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k == "structured" and v is not None:
            out[k] = _serialize_structured(v)
        elif k == "triage" and hasattr(v, "name"):
            out[k] = v.name
        else:
            out[k] = v
    return out


class InitializeRequest(BaseModel):
    preset: str = Field(
        default="who-malaria",
        description="who-malaria | uganda",
    )
    pdf_path: Optional[str] = None
    output_dir: Optional[str] = None
    reuse_existing_kb: bool = Field(default=True)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    full_response: bool = Field(default=True)
    use_local_llm: Optional[bool] = None
    local_llm_gguf: Optional[str] = None


def _local_llm_disk_configured() -> bool:
    p = (os.environ.get("SAFEAI_LLM_GGUF") or "").strip()
    return bool(p and os.path.isfile(p))


def _allow_client_supplied_gguf_path(request: Request) -> bool:
    v = os.environ.get("SAFEAI_ALLOW_CLIENT_GGUF_PATH", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    c = request.client
    host = (c.host if c else "") or ""
    return host in ("127.0.0.1", "::1")


def _local_llm_client_path_env_unlocked() -> bool:
    return os.environ.get("SAFEAI_ALLOW_CLIENT_GGUF_PATH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@app.get("/")
def root() -> Dict[str, Any]:
    """Service discovery (browser-friendly)."""
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "metadata": "/metadata",
        "initialize": "POST /initialize",
        "ask": "POST /ask",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pipeline": "v2",
        "vht_llm_default": "search_first",
        "vht_llm_legacy_env": "SAFEAI_VHT_LLM_LEGACY=1",
        "initialized": _qa is not None,
        "preset": (_loaded or {}).get("preset"),
        "document_title": (_loaded or {}).get("document_title"),
        "local_llm_gguf_configured": _local_llm_disk_configured(),
        "local_llm_env_enabled": os.environ.get("SAFEAI_USE_LOCAL_LLM", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "local_llm_client_path_env_unlocked": _local_llm_client_path_env_unlocked(),
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    if _qa is None or _loaded is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. POST /initialize first.")
    kb = os.path.join(_qa.output_dir, "knowledge_base.json")
    summ: Dict[str, Any] = {}
    if os.path.isfile(kb):
        import json

        with open(kb, "r", encoding="utf-8") as f:
            data = json.load(f)
        summ = data.get("extraction_summary", {})
    val = _qa.validation_result or {}
    overall = val.get("overall", {}) if isinstance(val, dict) else {}
    return {
        "preset": _loaded.get("preset"),
        "pdf_path": _loaded.get("pdf_path"),
        "output_dir": _loaded.get("output_dir"),
        "document_title": _loaded.get("document_title"),
        "chunk_count": len(_qa.chunks or []),
        "knowledge_base_path": kb,
        "extraction_summary": summ,
        "validation_overall": overall,
    }


@app.post("/initialize")
def initialize(req: InitializeRequest) -> Dict[str, Any]:
    global _qa, _loaded

    preset = _normalize_preset(req.preset)
    cfg = _build_config(preset, req.pdf_path, req.output_dir)

    if not os.path.isfile(cfg.pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF not found: {cfg.pdf_path}")

    kb_file = os.path.join(cfg.output_dir, "knowledge_base.json")
    if not req.reuse_existing_kb and os.path.isfile(kb_file):
        try:
            os.remove(kb_file)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not remove existing KB for rebuild: {e!s}")

    try:
        qa = MedicalQASystem(config=cfg)
        qa.initialize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {e!s}")

    _qa = qa
    _loaded = {
        "preset": preset,
        "pdf_path": cfg.pdf_path,
        "output_dir": cfg.output_dir,
        "document_title": cfg.document_title,
        "reuse_existing_kb": req.reuse_existing_kb,
    }

    return {
        "message": "Pipeline v2 initialized successfully",
        "config": _loaded,
        "chunk_count": len(qa.chunks or []),
    }


@app.post("/ask")
def ask(req: AskRequest, request: Request) -> Dict[str, Any]:
    if _qa is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized. POST /initialize first.")

    try:
        if req.full_response:
            allow_gguf = _allow_client_supplied_gguf_path(request)
            raw = _qa.answer_with_response(
                req.query,
                use_local_llm=req.use_local_llm,
                local_llm_gguf=(req.local_llm_gguf or "").strip() or None,
                allow_client_supplied_gguf_path=allow_gguf,
            )
            result = _serialize_ask_result(raw)
        else:
            result = _qa.answer(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {e!s}")

    return {
        "query": req.query,
        "full_response": req.full_response,
        "pipeline": "v2",
        "result": result,
    }
