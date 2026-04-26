"""
Medical pipeline package (capstone2 / pipeline v2).

Multi-pass extraction, validation, chunking, BM25 Q&A, guardrails, and VHT output
with search-first local LLM synthesis by default (see ``orchestrator``).

Heavy dependencies (PyMuPDF, rank_bm25, etc.) load only when you import
orchestrator / extractor / chunker. Config and presets load without them.
"""

from __future__ import annotations

import importlib
from typing import Any, List

from .config import (
    ExtractionConfig,
    ValidationReport,
    TriageLevel,
    DangerSign,
    MedicalSource,
    medical_source_for_config,
    DEFAULT_UGANDA_CLINICAL_2023_PDF,
    DEFAULT_WHO_MALARIA_NIH_PDF,
    GENERAL_CLINICAL_CRITICAL_TERMS,
    MALARIA_GUIDELINE_CRITICAL_TERMS,
    UGANDA_CLINICAL_CRITICAL_TERMS,
    extraction_config_who_malaria_nih,
    extraction_config_uganda_clinical_2023,
)

__all__: List[str] = [
    "ExtractionConfig",
    "ValidationReport",
    "TriageLevel",
    "DangerSign",
    "MedicalSource",
    "medical_source_for_config",
    "DEFAULT_WHO_MALARIA_NIH_PDF",
    "DEFAULT_UGANDA_CLINICAL_2023_PDF",
    "GENERAL_CLINICAL_CRITICAL_TERMS",
    "MALARIA_GUIDELINE_CRITICAL_TERMS",
    "UGANDA_CLINICAL_CRITICAL_TERMS",
    "extraction_config_who_malaria_nih",
    "extraction_config_uganda_clinical_2023",
    "MultiPassExtractor",
    "ExtractionValidator",
    "SmartChunker",
    "MedicalGuardrailBrain",
    "MedicalQASystem",
    "ResponseOrchestrator",
    "ResponseContent",
    "ResponseFormat",
    "infer_triage_from_query",
    "infer_query_intent",
]

_LAZY_EXPORTS = {
    "MultiPassExtractor": ("extractor", "MultiPassExtractor"),
    "ExtractionValidator": ("validator", "ExtractionValidator"),
    "SmartChunker": ("chunker", "SmartChunker"),
    "MedicalGuardrailBrain": ("guardrail", "MedicalGuardrailBrain"),
    "MedicalQASystem": ("orchestrator", "MedicalQASystem"),
    "ResponseOrchestrator": ("response", "ResponseOrchestrator"),
    "ResponseContent": ("response", "ResponseContent"),
    "ResponseFormat": ("response", "ResponseFormat"),
    "infer_triage_from_query": ("response", "infer_triage_from_query"),
    "infer_query_intent": ("retrieval", "infer_query_intent"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(f"{__name__}.{mod}")
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(__all__))
