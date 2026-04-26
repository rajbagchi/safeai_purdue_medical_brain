"""
Orchestrator: runs extraction, validation, chunking, and Q&A with guardrails.
"""

import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dataclasses import asdict

from .config import ExtractionConfig, TriageLevel
from .extractor import MultiPassExtractor
from .validator import ExtractionValidator
from .chunker import SmartChunker
from .guardrail import MedicalGuardrailBrain
from .config import medical_source_for_config
from .local_simplifier import (
    LocalSimplifierLLM,
    resolve_local_llm_for_ask,
    structured_answer_from_content,
)
from .search_first_llm import SearchFirstVHTLLM, normalize_triage_heading_for_guardrail
from .retrieval import infer_query_intent, retrieve_top_chunk_indices
from .response import (
    ResponseFormat,
    ResponseOrchestrator,
    infer_triage_from_query,
    severe_malaria_care_context,
)


def _vht_retrieval_pool_k() -> int:
    """BM25 pool for VHT + local LLM (pipeline v2 defaults wider than v1)."""
    return max(5, int(os.environ.get("SAFEAI_V2_RETRIEVAL_K", "18")))


def _vht_llm_use_legacy_simplifier() -> bool:
    v = os.environ.get("SAFEAI_VHT_LLM_LEGACY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class MedicalQASystem:
    """
    Complete medical Q&A system with:
    - Multi-pass extraction
    - Comprehensive validation
    - Smart chunking
    - Two-brain guardrails
    - Pipeline v2: local VHT uses search-first LLM synthesis (see ``answer_with_response``)
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        output_dir: str = "./medical_knowledge_base",
        *,
        config: Optional[ExtractionConfig] = None,
    ):
        if config is not None:
            if pdf_path is not None:
                raise ValueError("Pass either config=... or pdf_path=..., not both.")
            self.config = config
        elif pdf_path is not None:
            self.config = ExtractionConfig(
                pdf_path=pdf_path,
                output_dir=output_dir,
                cache_dir=os.path.join(output_dir, "cache"),
            )
        else:
            raise ValueError("MedicalQASystem requires pdf_path=... or config=...")

        self.pdf_path = self.config.pdf_path
        self.output_dir = self.config.output_dir

        self.extraction_result: Dict[str, Any] | None = None
        self.validation_result: Dict[str, Any] | None = None
        self.chunks: List[Dict] | None = None
        self.search_index: Dict[str, Any] | None = None
        self.guardrail: MedicalGuardrailBrain | None = None
        self._response_orchestrator: Optional[ResponseOrchestrator] = None

    def initialize(self) -> "MedicalQASystem":
        """Initialize or load existing knowledge base."""
        from .compat import fix_stdio_encoding

        fix_stdio_encoding()
        print("=" * 70)
        print("MEDICAL Q&A SYSTEM - COMPLETE PIPELINE")
        print("=" * 70)

        kb_file = os.path.join(self.output_dir, "knowledge_base.json")

        if os.path.exists(kb_file):
            print("\n📦 Loading existing knowledge base...")
            self._load_knowledge_base(kb_file)
        else:
            print("\n🔄 Building new knowledge base...")
            self._build_knowledge_base()

        print("\n✅ System ready!")
        return self

    def _build_knowledge_base(self) -> None:
        """Build knowledge base from scratch."""
        print("\n" + "=" * 70)
        print("STEP 1: MULTI-PASS EXTRACTION")
        print("=" * 70)

        extractor = MultiPassExtractor(self.config)
        self.extraction_result = extractor.extract_all()

        print("\n" + "=" * 70)
        print("STEP 2: VALIDATION")
        print("=" * 70)

        validator = ExtractionValidator(
            self.extraction_result,
            self.config,
        )
        self.validation_result = validator.validate_all()

        print("\n" + "=" * 70)
        print("STEP 3: SMART CHUNKING")
        print("=" * 70)

        chunker = SmartChunker(self.extraction_result, self.config)
        self.chunks = chunker.chunk_by_headings()
        self.search_index = chunker.create_search_index()

        print("\n" + "=" * 70)
        print("STEP 4: INITIALIZING GUARDRAIL BRAIN")
        print("=" * 70)

        self.guardrail = MedicalGuardrailBrain(self.chunks)

        self._save_knowledge_base()

    def _save_knowledge_base(self) -> None:
        """Save knowledge base to disk."""
        assert self.extraction_result is not None
        assert self.validation_result is not None
        assert self.chunks is not None

        kb = {
            "metadata": {
                "pdf_path": self.pdf_path,
                "build_date": datetime.now().isoformat(),
                "config": asdict(self.config),
                "validation": {
                    k: (
                        asdict(v)
                        if hasattr(v, "__dataclass_fields__")
                        else v
                    )
                    for k, v in self.validation_result.items()
                },
            },
            "chunks": self.chunks,
            "extraction_summary": {
                "pages": len(self.extraction_result.get("pages", [])),
                "tables": len(self.extraction_result.get("tables", [])),
                "images": len(self.extraction_result.get("images", [])),
                "passes": len(
                    self.extraction_result.get("extraction_log", [])
                ),
            },
        }

        kb_file = os.path.join(self.output_dir, "knowledge_base.json")
        with open(kb_file, "w") as f:
            json.dump(kb, f, indent=2, default=str)

        chunks_file = os.path.join(self.output_dir, "chunks.json")
        with open(chunks_file, "w") as f:
            json.dump(self.chunks, f, indent=2, default=str)

        print(f"\n💾 Knowledge base saved to {self.output_dir}")

    def _load_knowledge_base(self, kb_file: str) -> None:
        """Load existing knowledge base."""
        with open(kb_file, "r") as f:
            kb = json.load(f)

        chunks_file = os.path.join(self.output_dir, "chunks.json")
        with open(chunks_file, "r") as f:
            self.chunks = json.load(f)

        self.validation_result = kb["metadata"].get("validation", {})

        chunker = SmartChunker({}, self.config)
        chunker.chunks = self.chunks
        self.search_index = chunker.create_search_index()

        self.guardrail = MedicalGuardrailBrain(self.chunks)

        print(f"\n📚 Loaded {len(self.chunks)} chunks")
        overall = kb["metadata"]["validation"].get("overall", {})
        conf = overall.get("confidence", 0)
        print(f"📊 Validation confidence: {conf:.1%}")

    def get_extraction_summary_from_disk(self) -> Dict[str, Any]:
        """Summary written to knowledge_base.json (pages/tables/passes counts)."""
        kb_path = os.path.join(self.output_dir, "knowledge_base.json")
        if not os.path.isfile(kb_path):
            return {}
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        return kb.get("extraction_summary", {})

    def _guardrail_evidence_footer(self, sources: List[Dict], query: str) -> str:
        """
        BM25 answers are evidence excerpts. Append sections required by
        MedicalGuardrailBrain. Triage line uses infer_triage_from_query (RED /
        YELLOW / GREEN) so it matches the response layer.
        """
        from .config import TriageLevel as TL

        pages = sorted({int(s["page"]) for s in sources if "page" in s})
        pages_str = ", ".join(f"Page {p}" for p in pages) if pages else "N/A"
        level, _reasons = infer_triage_from_query(query)
        if level == TL.RED:
            triage = (
                "Triage Level: RED (query may indicate danger signs — urgent "
                "assessment and referral per local protocol; excerpts are supportive "
                "information only)\n\n"
            )
        elif level == TL.YELLOW:
            triage = (
                "Triage Level: YELLOW (time-sensitive symptoms — assess at health "
                "facility today per local protocol; excerpts are supportive only)\n\n"
            )
        else:
            triage = (
                "Triage Level: GREEN (evidence retrieval summary — not a substitute "
                "for bedside assessment; follow local protocols)\n\n"
            )
        return (
            "\n---\n\n"
            + triage
            + "Immediate Actions: Review the guideline excerpts above; align actions "
            "with national/WHO guidance and qualified supervision.\n\n"
            "Next Steps / Monitoring: Consult the full source guideline or a "
            "clinician for patient-specific decisions.\n\n"
            "When to Refer: Per excerpts and national guidance; seek urgent care "
            "if danger signs, severe disease, or instability is suspected.\n\n"
            f"Citations: {pages_str}\n"
        )

    def _retrieve_top_k(self, query: str, k: int = 5) -> tuple[List[int], List[Dict], List[Dict]]:
        """BM25 over a wide pool, re-ranked by heading quality; returns indices, sources, chunk dicts."""
        assert self.chunks is not None
        assert self.search_index is not None
        query_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        query_tokens = [t for t in query_tokens if len(t) > 1]
        bm25 = self.search_index["bm25"]
        top_indices = retrieve_top_chunk_indices(
            bm25,
            self.chunks,
            query_tokens,
            k=k,
        )
        sources: List[Dict] = []
        chunk_dicts: List[Dict] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            sources.append({"page": chunk["page"], "heading": chunk.get("heading", "")})
            chunk_dicts.append(dict(chunk))
        return list(top_indices), sources, chunk_dicts

    def answer(self, query: str) -> Dict:
        """Answer a medical query with guardrail validation."""
        assert self.guardrail is not None

        _, sources, chunks_top = self._retrieve_top_k(query, 5)

        response = f"**{self.config.document_title}**\n\n"
        response += f"**Question:** {query}\n\n"

        for i, (s, chunk) in enumerate(zip(sources, chunks_top), 1):
            response += f"### {i}. {s['heading']}\n\n"
            response += chunk["text"][:500] + "...\n\n"
            response += f"📄 **Reference:** Page {s['page']}\n\n"

        response += self._guardrail_evidence_footer(sources, query)

        validation = self.guardrail.validate_response(query, response)

        if not validation["passed"] or validation["warnings"]:
            response += "\n---\n"
            response += "**🧪 Guardrail Brain Validation:**\n\n"

            if validation["errors"]:
                response += "**❌ SAFETY ERRORS - DO NOT USE:**\n"
                for e in validation["errors"]:
                    response += f"• ⚠️ {e}\n"

            if validation["warnings"]:
                response += "**⚠️ Warnings:**\n"
                for w in validation["warnings"]:
                    response += f"• {w}\n"
        elif validation["passed"]:
            response += (
                "\n---\n**🧪 Guardrail Brain Validation:** ✅ Passed\n"
            )

        return {
            "query": query,
            "response": response,
            "sources": sources,
            "validation": validation,
            "validation_passed": validation["passed"],
        }

    def _response_orch(self) -> ResponseOrchestrator:
        if self._response_orchestrator is None:
            self._response_orchestrator = ResponseOrchestrator()
        return self._response_orchestrator

    def answer_with_response(
        self,
        query: str,
        *,
        use_local_llm: Optional[bool] = None,
        local_llm_gguf: Optional[str] = None,
        allow_client_supplied_gguf_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline output: BM25 + guardrail + VHT response layer (standard,
        quick, referral note) + structured ResponseContent.

        Local LLM (when enabled): search-first synthesis from retrieval excerpts
        (set ``SAFEAI_VHT_LLM_LEGACY=1`` to use the v1 draft-rewrite simplifier instead).
        """
        assert self.guardrail is not None

        pool_k = _vht_retrieval_pool_k()
        _, sources, retrieved_chunks = self._retrieve_top_k(query, pool_k)
        evidence_sources = sources[:5]
        evidence_chunks = retrieved_chunks[:5]

        response = f"**{self.config.document_title}**\n\n"
        response += f"**Question:** {query}\n\n"
        for i, (s, chunk) in enumerate(zip(evidence_sources, evidence_chunks), 1):
            response += f"### {i}. {s['heading']}\n\n"
            response += chunk["text"][:500] + "...\n\n"
            response += f"📄 **Reference:** Page {s['page']}\n\n"

        response += self._guardrail_evidence_footer(evidence_sources, query)
        validation = self.guardrail.validate_response(query, response)

        if not validation["passed"] or validation["warnings"]:
            response += "\n---\n**🧪 Guardrail Brain Validation:**\n\n"
            if validation["errors"]:
                response += "**❌ SAFETY ERRORS - DO NOT USE:**\n"
                for e in validation["errors"]:
                    response += f"• ⚠️ {e}\n"
            if validation["warnings"]:
                response += "**⚠️ Warnings:**\n"
                for w in validation["warnings"]:
                    response += f"• {w}\n"
        elif validation["passed"]:
            response += "\n---\n**🧪 Guardrail Brain Validation:** ✅ Passed\n"

        intent = infer_query_intent(query)
        triage, triage_reasons = infer_triage_from_query(query)
        if triage == TriageLevel.GREEN and severe_malaria_care_context(
            query, retrieved_chunks
        ):
            triage = TriageLevel.YELLOW
            triage_reasons = [
                "Retrieved passages focus on severe or cerebral malaria — "
                "follow hospital-level protocols, not routine home care alone.",
            ]
        med_src = medical_source_for_config(self.config)
        orch = self._response_orch()
        structured = orch.create(
            query=query,
            triage=triage,
            triage_reasons=triage_reasons,
            guardrail_output=validation,
            retrieved_chunks=retrieved_chunks,
            source=med_src,
            dosage_info=None,
            query_intent=intent,
        )

        rule_vht = structured.to_vht_format()
        referral_note = orch.formatter.format(structured, ResponseFormat.REFERRAL)
        quick_summary = orch.formatter.format(structured, ResponseFormat.VHT_QUICK)

        vht_response = rule_vht
        local_llm_used = False
        local_llm_skipped_reason: Optional[str] = None
        vht_synthesis_mode = "rule_based_only"

        run_llm, llm_path, skip_llm = resolve_local_llm_for_ask(
            use_local_llm,
            client_gguf_path=local_llm_gguf,
            accept_client_path=allow_client_supplied_gguf_path,
        )
        if not run_llm or not llm_path:
            local_llm_skipped_reason = skip_llm
        else:
            packet = structured_answer_from_content(
                structured,
                query=query,
                document_title=self.config.document_title,
            )
            candidate: Optional[str] = None
            llm_internal_err: Optional[str] = None
            legacy = _vht_llm_use_legacy_simplifier()
            try:
                if legacy:
                    vht_synthesis_mode = "legacy_simplifier"
                    sim = LocalSimplifierLLM(model_path=llm_path)
                    candidate, llm_internal_err = sim.simplify_vht_markdown(
                        query=query,
                        document_title=self.config.document_title,
                        rule_based_vht=rule_vht,
                        structured_answer=packet,
                        evidence_chunks=retrieved_chunks,
                    )
                else:
                    vht_synthesis_mode = "search_first_llm"
                    sim2 = SearchFirstVHTLLM(model_path=llm_path)
                    candidate, llm_internal_err = sim2.synthesize(
                        query=query,
                        document_title=self.config.document_title,
                        approved_structured_block=packet,
                        evidence_chunks=retrieved_chunks,
                    )
            except Exception as e:
                candidate = None
                llm_internal_err = f"local_llm_error: {e!s}"[:240]
                local_llm_skipped_reason = llm_internal_err

            if candidate:
                if not legacy:
                    candidate = normalize_triage_heading_for_guardrail(candidate)
                v2 = self.guardrail.validate_response(query, candidate)
                if v2.get("passed"):
                    vht_response = candidate
                    local_llm_used = True
                else:
                    errs = v2.get("errors") or []
                    local_llm_skipped_reason = (
                        "guardrail_reject_llm: " + "; ".join(str(x) for x in errs[:4])
                    )
            elif llm_internal_err:
                local_llm_skipped_reason = llm_internal_err[:240]
            elif local_llm_skipped_reason is None:
                local_llm_skipped_reason = "local_llm_no_output"

        return {
            "query": query,
            "query_intent": intent,
            "response": response,
            "sources": evidence_sources,
            "validation": validation,
            "validation_passed": validation["passed"],
            "triage": triage,
            "triage_reasons": triage_reasons,
            "vht_response": vht_response,
            "referral_note": referral_note,
            "quick_summary": quick_summary,
            "structured": structured,
            "local_llm_used": local_llm_used,
            "local_llm_skipped_reason": local_llm_skipped_reason,
            "vht_synthesis_mode": vht_synthesis_mode,
            "vht_retrieval_pool_k": pool_k,
        }
