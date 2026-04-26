"""
Pipeline configuration, enums, and shared data structures.

ExtractionConfig accepts any PDF path string (Windows absolute paths, paths with
spaces, etc.). Use the factory helpers for the two validated source documents
so output directories do not collide and medical-content checks match the doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# This repo root (parent of ``pipeline/``) — default KB dirs live here, separate from pipeline v1.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class TriageLevel(Enum):
    RED = "🔴 RED (Immediate Referral Required)"
    YELLOW = "🟡 YELLOW (Urgent Referral - Assess Today)"
    GREEN = "🟢 GREEN (Manage at Community Level)"


class MedicalSource(Enum):
    """Canonical label for citation / VHT output (maps from active guideline preset)."""

    WHO_MALARIA_NIH = "WHO Malaria Guidelines (NCBI Bookshelf)"
    UGANDA_CLINICAL_2023 = "Uganda Clinical Guidelines 2023"
    GENERIC = "Clinical guidelines"


def medical_source_for_config(cfg: "ExtractionConfig") -> MedicalSource:
    """Infer MedicalSource from ExtractionConfig.document_title."""
    t = (cfg.document_title or "").lower()
    if "uganda" in t:
        return MedicalSource.UGANDA_CLINICAL_2023
    if "malaria" in t or "ncbi" in t or "bookshelf" in t:
        return MedicalSource.WHO_MALARIA_NIH
    return MedicalSource.GENERIC


class DangerSign(Enum):
    # Pediatric danger signs (under 5)
    UNABLE_TO_DRINK = "Unable to drink or breastfeed"
    CONVULSIONS = "Convulsions"
    VOMITS_EVERYTHING = "Vomits everything"
    LETHARGIC = "Lethargic or unconscious"

    # Adult danger signs
    CHEST_PAIN = "Chest pain"
    DIFFICULT_BREATHING = "Difficulty breathing"
    SEVERE_HEADACHE = "Severe headache with neck stiffness"
    BLEEDING = "Unexplained bleeding"

    # General
    HIGH_FEVER = "High fever (>39°C) for >3 days"
    DEHYDRATION = "Signs of severe dehydration"


# Terms searched in extracted text for Stage 4 (medical plausibility heuristic).
# WHO malaria PDF is expected to match MALARIA_*; Uganda broad guidelines use GENERAL_*.
# Default locations for validated source PDFs (override with --pdf / env as needed).
DEFAULT_WHO_MALARIA_NIH_PDF = Path(r"C:\temp\capstone\Bookshelf_NBK588130.pdf")
DEFAULT_UGANDA_CLINICAL_2023_PDF = Path(
    r"C:\temp\capstone\Uganda Clinical Guidelines 2023.pdf"
)

GENERAL_CLINICAL_CRITICAL_TERMS: List[str] = [
    "dose",
    "mg",
    "treatment",
    "patient",
    "child",
    "adult",
    "contraindication",
    "referral",
    "emergency",
    "management",
]

MALARIA_GUIDELINE_CRITICAL_TERMS: List[str] = [
    "dose",
    "mg",
    "contraindication",
    "warning",
    "severe malaria",
    "artemisinin",
    "pregnancy",
    "children",
    "infant",
    "emergency",
]

# Slightly tuned for MoH / broad specialty content (avoid malaria-only terms).
UGANDA_CLINICAL_CRITICAL_TERMS: List[str] = [
    "dose",
    "mg",
    "treatment",
    "diagnosis",
    "patient",
    "child",
    "adult",
    "contraindication",
    "referral",
    "hospital",
    "symptom",
]


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""

    pdf_path: str
    output_dir: str = "./medical_knowledge_base"
    cache_dir: str = "./cache"
    enable_ocr: bool = True
    enable_table_detection: bool = True
    enable_image_extraction: bool = True
    # Scan every page for PyMuPDF tables (not only the Pass-0 sample); slower but catches tables after page ~20.
    full_document_table_scan: bool = True
    # Bump to invalidate extraction pickle cache after engine changes.
    extraction_engine_version: int = 2
    min_chunk_size: int = 500
    max_chunk_size: int = 2000
    chunk_overlap: int = 200
    confidence_threshold: float = 0.8
    num_extraction_passes: int = 3
    # Shown in Q&A responses and saved metadata
    document_title: str = "Clinical guidelines"
    # If None, validator uses GENERAL_CLINICAL_CRITICAL_TERMS
    critical_content_terms: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # Normalize path: expanduser, resolve for stable cache keys on absolute paths
        p = Path(self.pdf_path).expanduser()
        try:
            self.pdf_path = str(p.resolve(strict=False))
        except OSError:
            self.pdf_path = str(p)
        if not self.cache_dir or self.cache_dir == "./cache":
            self.cache_dir = str(Path(self.output_dir) / "cache")


def extraction_config_who_malaria_nih(
    pdf_path: str | Path = Path(r"C:\temp\capstone\Bookshelf_NBK588130.pdf"),
    *,
    output_dir: Optional[str | Path] = None,
) -> ExtractionConfig:
    """
    Preset for Bookshelf NIH WHO malaria booklet (NCBI NBK588130).
    """
    pdf = Path(pdf_path).expanduser()
    out = Path(output_dir) if output_dir else _REPO_ROOT / "kb_who_malaria"
    return ExtractionConfig(
        pdf_path=str(pdf),
        output_dir=str(out),
        document_title="WHO Malaria Guidelines (NCBI Bookshelf)",
        critical_content_terms=list(MALARIA_GUIDELINE_CRITICAL_TERMS),
    )


def extraction_config_uganda_clinical_2023(
    pdf_path: str | Path = DEFAULT_UGANDA_CLINICAL_2023_PDF,
    *,
    output_dir: Optional[str | Path] = None,
) -> ExtractionConfig:
    """
    Preset for Uganda Clinical Guidelines 2023 (broad MoH content).
    """
    pdf = Path(pdf_path).expanduser()
    out = Path(output_dir) if output_dir else _REPO_ROOT / "kb_uganda_clinical_2023"
    return ExtractionConfig(
        pdf_path=str(pdf),
        output_dir=str(out),
        document_title="Uganda Clinical Guidelines 2023",
        critical_content_terms=list(UGANDA_CLINICAL_CRITICAL_TERMS),
    )


@dataclass
class ValidationReport:
    """Report from validation stages."""

    stage: str
    passed: bool
    issues: List[str]
    confidence: float
    suggestions: List[str]
    metadata: Dict[str, Any]
