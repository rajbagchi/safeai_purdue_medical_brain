"""
Local GGUF simplifier (e.g. Qwen2.5-3B-Instruct) via llama-cpp-python.

Rewrites the rule-based VHT draft using the approved structured answer, retrieved
excerpts (primary for clinical specifics), and limited general knowledge for
plain-language framing only. Output is re-validated with MedicalGuardrailBrain
before use.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from .config import MedicalSource, TriageLevel

_LOCK = threading.Lock()
_LLAMA = None
_LLAMA_PATH: Optional[str] = None


def local_llm_enabled(explicit: Optional[bool] = None) -> bool:
    """
    ``explicit`` overrides env: ``False`` off; ``True`` on only if GGUF path exists;
    ``None`` uses ``SAFEAI_USE_LOCAL_LLM`` plus path.
    """
    path = (os.environ.get("SAFEAI_LLM_GGUF") or "").strip()
    if not path or not os.path.isfile(path):
        return False if explicit is not True else False
    if explicit is False:
        return False
    if explicit is True:
        return True
    v = os.environ.get("SAFEAI_USE_LOCAL_LLM", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_valid_gguf_model_path(path: str) -> bool:
    """True if path is non-empty, ends with .gguf, and is a readable file."""
    p = (path or "").strip()
    if not p:
        return False
    try:
        p_abs = os.path.realpath(os.path.normpath(os.path.expanduser(p)))
    except OSError:
        return False
    if not p_abs.lower().endswith(".gguf"):
        return False
    return os.path.isfile(p_abs)


def resolve_local_llm_for_ask(
    use_local_llm: Optional[bool],
    *,
    client_gguf_path: Optional[str],
    accept_client_path: bool,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Decide whether to run the local simplifier and which file to load.

    Returns ``(run, model_path, skipped_reason)``. ``skipped_reason`` is ``None``
    when ``run`` is True; otherwise a short machine-readable explanation.
    """
    env_raw = (os.environ.get("SAFEAI_LLM_GGUF") or "").strip()
    resolved: Optional[str] = None

    if accept_client_path and client_gguf_path:
        c = client_gguf_path.strip()
        if c and is_valid_gguf_model_path(c):
            resolved = os.path.realpath(os.path.normpath(os.path.expanduser(c)))

    if resolved is None and env_raw and os.path.isfile(env_raw):
        resolved = os.path.realpath(os.path.normpath(os.path.expanduser(env_raw)))

    if use_local_llm is False:
        return False, None, "local_llm_off_explicit_false_in_request"

    if resolved is None:
        if use_local_llm is True and (client_gguf_path or "").strip():
            if accept_client_path:
                return (
                    False,
                    None,
                    "local_llm_disabled_invalid_or_missing_client_gguf_path",
                )
            return (
                False,
                None,
                "local_llm_client_gguf_ignored_non_loopback_set_SAFEAI_ALLOW_CLIENT_GGUF_PATH",
            )
        return (
            False,
            None,
            "local_llm_disabled_no_valid_SAFEAI_LLM_GGUF_file_on_this_process",
        )

    if use_local_llm is True:
        return True, resolved, None

    v = os.environ.get("SAFEAI_USE_LOCAL_LLM", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True, resolved, None
    return (
        False,
        None,
        "local_llm_disabled_set_SAFEAI_USE_LOCAL_LLM_to_1_or_pass_use_local_llm_true",
    )


def _evidence_block(
    chunks: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
    total_max: Optional[int] = None,
) -> str:
    """Build excerpt text with per-chunk and total character caps (helps fit n_ctx)."""
    if max_chars is None:
        max_chars = int(os.environ.get("SAFEAI_LLM_EVIDENCE_CHUNK_CHARS", "1000"))
    if total_max is None:
        total_max = int(os.environ.get("SAFEAI_LLM_EVIDENCE_TOTAL_CHARS", "4800"))
    parts: List[str] = []
    budget = total_max
    for i, ch in enumerate(chunks, 1):
        if budget <= 0:
            break
        h = str(ch.get("heading", "")).strip()
        p = ch.get("page", "?")
        take = min(max_chars, max(0, budget - 48))
        t = (ch.get("text") or "")[:take]
        piece = f"[{i}] Page {p} | {h}\n{t}"
        parts.append(piece)
        budget -= len(piece) + 2
    return "\n\n".join(parts)


def structured_answer_for_prompt(
    *,
    query: str,
    document_title: str,
    triage: TriageLevel,
    triage_reasons: List[str],
    actions: List[str],
    monitoring: List[str],
    referral_criteria: List[str],
    citations: List[Dict[str, Any]],
    family_message: Optional[str],
    danger_signs: Optional[List[str]] = None,
) -> str:
    """Deterministic text the model must not contradict (facts + triage intent)."""
    tri = triage.name if hasattr(triage, "name") else str(triage)
    lines = [
        f"Question: {query}",
        f"Guideline title: {document_title}",
        f"Triage (approved): {tri}",
        f"Triage reasons: {', '.join(triage_reasons) if triage_reasons else 'n/a'}",
        "",
        "Approved actions:",
        *[f"- {a}" for a in actions],
        "",
        "Monitoring:",
        *[f"- {m}" for m in monitoring],
        "",
        "Referral criteria:",
        *[f"- {r}" for r in referral_criteria],
    ]
    if danger_signs:
        lines.extend(["", "Danger signs (from guideline metadata):", *[f"- {d}" for d in danger_signs]])
    lines.extend(
        [
        "",
        "Citations (from retrieval):",
    ]
    )
    for c in citations[:8]:
        src = c.get("source", "")
        if isinstance(src, MedicalSource):
            src = src.value
        lines.append(f"- {src} Page {c.get('page', '?')}: {c.get('section', '')}")
    if family_message:
        lines.extend(["", "Family-facing line (approved):", family_message])
    return "\n".join(lines)


SYSTEM_PROMPT = """You rewrite material for Village Health Team (VHT) workers with low literacy.

Grounding (most important):
- Answer the user's question using the Evidence excerpts first. If a drug, regimen, duration, or dose appears there, you may state it clearly in Immediate Actions and tie it to that source (page or [n]).
- The approved structured block (triage, bullets, citations list) must not be contradicted: keep the same RED, YELLOW, or GREEN meaning and the same urgency of referral.
- Specific treatments (which medicine, dose, schedule) must be supported by Evidence excerpts and/or the approved action/monitoring/referral bullets. If excerpts are off-topic or silent on the question, say that the retrieved pages do not spell out the regimen here and point to supervision/national protocol—do not invent doses.

General knowledge (allowed, narrow):
- You may use prior medical knowledge only for plain-language framing: definitions, why adherence matters, how to explain risk without new facts, or organizing excerpt-supported points. Do not introduce drugs, doses, or diagnoses that are not in the excerpts or approved bullets.

Style:
- Short, simple sentences. Prefer bullet lines. Do not use emoji.
- If danger signs apply (per approved triage), put urgent referral language first.

You MUST output plain markdown using exactly these section headings in this order (each heading on its own line, then content):
Triage Level:
Immediate Actions:
Next Steps / Monitoring:
When to Refer:
Citations:

Under "Triage Level:" write the level (RED, YELLOW, or GREEN) and a one-line plain reason matching the approved triage.
Under "Citations:" list relevant pages/sections from the evidence (e.g. Page 96)."""


def _load_llama(model_path: str):
    global _LLAMA, _LLAMA_PATH

    from llama_cpp import Llama  # type: ignore[import-not-found]

    with _LOCK:
        if _LLAMA is not None and _LLAMA_PATH == model_path:
            return _LLAMA
        n_ctx = int(os.environ.get("SAFEAI_LLM_N_CTX", "8192"))
        n_threads = int(os.environ.get("SAFEAI_LLM_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
        n_gpu = int(os.environ.get("SAFEAI_LLM_N_GPU_LAYERS", "0"))
        _LLAMA = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu,
            verbose=os.environ.get("SAFEAI_LLM_VERBOSE", "").strip().lower() in ("1", "true", "yes"),
        )
        _LLAMA_PATH = model_path
        return _LLAMA


def structured_answer_from_content(
    structured: Any,
    *,
    query: str,
    document_title: str,
) -> str:
    """Build the approved structured packet from ``ResponseContent``."""
    return structured_answer_for_prompt(
        query=query,
        document_title=document_title,
        triage=structured.triage,
        triage_reasons=structured.triage_reasons,
        actions=structured.actions,
        monitoring=structured.monitoring,
        referral_criteria=structured.referral_criteria,
        citations=structured.citations,
        family_message=structured.family_message,
        danger_signs=getattr(structured, "danger_signs", None) or None,
    )


class LocalSimplifierLLM:
    """
    Controlled rewriting: clearer VHT text grounded in retrieval + approved packet,
    with narrow use of general knowledge for framing only.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = (model_path or os.environ.get("SAFEAI_LLM_GGUF") or "").strip()

    @property
    def available(self) -> bool:
        return bool(self.model_path) and os.path.isfile(self.model_path)

    def simplify_vht_markdown(
        self,
        *,
        query: str,
        document_title: str,
        rule_based_vht: str,
        structured_answer: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns ``(markdown, error)``. ``error`` is set when ``markdown`` is ``None`` so
        callers can distinguish load/inference failures from guardrail rejection.
        """
        if not self.available:
            return None, "local_llm_unavailable_missing_model_file"

        draft_cap = int(os.environ.get("SAFEAI_LLM_RULE_DRAFT_CHARS", "4500"))
        user_max = int(os.environ.get("SAFEAI_LLM_MAX_USER_CHARS", "14000"))
        user = (
            f"{structured_answer}\n\n"
            "---\nRule-based draft (tone reference; excerpts override wrong lines here):\n"
            f"{rule_based_vht[:draft_cap]}\n\n"
            "---\nEvidence excerpts (primary source for treatment and clinical specifics):\n"
            f"{_evidence_block(evidence_chunks)}\n\n"
            "Rewrite for VHT readers: lead Immediate Actions with excerpt-supported steps that "
            "directly answer the user question; use general knowledge only as allowed in the "
            "system rules. Output markdown only."
        )
        prefix = f"User question:\n{query}\n\n"
        full_user = prefix + user
        if len(full_user) > user_max:
            full_user = full_user[:user_max]

        max_tokens = int(os.environ.get("SAFEAI_LLM_MAX_TOKENS", "2048"))
        temperature = float(os.environ.get("SAFEAI_LLM_TEMPERATURE", "0.15"))

        try:
            llm = _load_llama(self.model_path)
            with _LOCK:
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            choice0 = (out.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            text = msg.get("content")
        except Exception as e:
            return None, f"local_llm_error: {e!s}"[:500]

        text = (text or "").strip()
        if not text:
            finish_reason = None
            try:
                finish_reason = (out.get("choices") or [{}])[0].get("finish_reason")
            except Exception:
                pass
            detail = finish_reason or "empty_content"
            return None, f"local_llm_empty_model_response ({detail})"[:500]
        return text, None
