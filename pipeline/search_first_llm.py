"""
Search-first VHT synthesis: local LLM composes VHT markdown primarily from BM25
excerpts, constrained by the approved structured packet (same guardrail sections).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .local_simplifier import _LOCK, _load_llama


def evidence_block_for_vht(
    chunks: List[Dict[str, Any]],
    *,
    max_chars: int = 1400,
    total_max: int = 12000,
) -> str:
    parts: List[str] = []
    budget = total_max
    for i, ch in enumerate(chunks, 1):
        if budget <= 0:
            break
        h = str(ch.get("heading", "")).strip()
        p = ch.get("page", "?")
        take = min(max_chars, max(0, budget - 56))
        t = (ch.get("text") or "")[:take]
        piece = f"[Evidence {i}] Page {p} | {h}\n{t}"
        parts.append(piece)
        budget -= len(piece) + 2
    return "\n\n".join(parts)


SYSTEM_SEARCH_FIRST = """You write the final answer for Village Health Team (VHT) workers.

Your job is to produce the best, clearest VHT response using the numbered Evidence
passages as the primary source of clinical facts (drugs, doses, steps, referral).

Hard rules:
- Match the approved triage level and meaning exactly (RED / YELLOW / GREEN from the
  approved block). Do not downgrade urgency.
- Treatments, doses, schedules, and referral triggers must come from Evidence and/or
  the approved bullet lists. If Evidence is silent or off-topic, say the retrieved pages
  do not spell this out here and tell the VHT to follow national protocol and seek
  supervision — do not invent doses or drugs.
- Use short, simple sentences and bullet lines under each section. No emoji.

Output plain markdown with exactly these section headings in this order:
Triage Level:
Immediate Actions:
Next Steps / Monitoring:
When to Refer:
Citations:

Under Triage Level: state RED, YELLOW, or GREEN (as approved) plus one plain sentence why.
Under Citations: cite pages from Evidence (e.g. Page 12) and section titles when helpful.
"""


class SearchFirstVHTLLM:
    def __init__(self, model_path: str) -> None:
        self.model_path = (model_path or "").strip()

    @property
    def available(self) -> bool:
        return bool(self.model_path) and os.path.isfile(self.model_path)

    def synthesize(
        self,
        *,
        query: str,
        document_title: str,
        approved_structured_block: str,
        evidence_chunks: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.available:
            return None, "local_llm_unavailable_missing_model_file"

        max_chars = int(os.environ.get("SAFEAI_SEARCH_FIRST_CHUNK_CHARS", "1400"))
        total_max = int(os.environ.get("SAFEAI_SEARCH_FIRST_TOTAL_CHARS", "12000"))
        ev = evidence_block_for_vht(
            evidence_chunks,
            max_chars=max_chars,
            total_max=total_max,
        )
        user = (
            f"User question:\n{query}\n\n"
            f"Active guideline document title: {document_title}\n\n"
            "---\nApproved structured constraints (do not contradict):\n"
            f"{approved_structured_block}\n\n"
            "---\nEvidence passages (primary clinical source):\n"
            f"{ev}\n\n"
            "Write the VHT markdown sections. Lead Immediate Actions with steps that "
            "directly answer the question and are supported by Evidence. "
            "If a dose appears in Evidence, you may state it clearly and cite the page."
        )
        user_max = int(os.environ.get("SAFEAI_SEARCH_FIRST_USER_MAX_CHARS", "28000"))
        if len(user) > user_max:
            user = user[:user_max]

        max_tokens = int(os.environ.get("SAFEAI_LLM_MAX_TOKENS", "2048"))
        temperature = float(os.environ.get("SAFEAI_SEARCH_FIRST_TEMPERATURE", "0.12"))

        try:
            llm = _load_llama(self.model_path)
            with _LOCK:
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_SEARCH_FIRST},
                        {"role": "user", "content": user},
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
            return None, "local_llm_empty_model_response"
        return text, None


def normalize_triage_heading_for_guardrail(text: str) -> str:
    if not text:
        return text
    t = text
    if re.search(r"(?im)^\s*Triage Level\s*:", t):
        return t
    m = re.search(
        r"(?im)^\s*(?:#+\s*)?\*{0,2}\s*Triage\s*:?\s*\*{0,2}\s*(RED|YELLOW|GREEN)",
        t,
        re.MULTILINE,
    )
    if m:
        color = m.group(1)
        rest = t[m.end() :].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        line2 = f"Triage Level: {color}"
        if rest and not rest.startswith("\n"):
            line2 += " — " + rest.split("\n", 1)[0].strip()
        block = m.group(0)
        t = t.replace(block, line2, 1)
    return t
