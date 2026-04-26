"""
Retrieval helpers: query intent for routing, BM25 pool + quality-weighted re-ranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

_REF_WORD = re.compile(r"\b(refer|referral|referred|referring)\b", re.IGNORECASE)
_PATIENT_WORD = re.compile(r"\bpatient\b", re.IGNORECASE)

# Headings that are usually PDF/table artifacts, not real section titles.
_JUNK_HEADINGS = frozenset({
    "h",
    "high",
    "low",
    "medium",
    "yes",
    "no",
    "n/a",
    "na",
    "tbd",
    "iv",
    "im",
    "po",
    "sc",
})


def infer_query_intent(query: str) -> str:
    """
    Lightweight intent for VHT messaging and triage hints.

    Returns one of: ``referral_hospital``, ``dosing``, ``general``.
    """
    q = (query or "").lower()
    has_ref_word = bool(_REF_WORD.search(q))

    if "when to refer" in q or "when should i refer" in q or "when do i refer" in q:
        return "referral_hospital"
    if "refer to hospital" in q or "referral to hospital" in q or "hospital referral" in q:
        return "referral_hospital"
    if "hospital" in q and (
        has_ref_word
        or "send to" in q
        or "send patient" in q
        or "admission" in q
        or re.search(r"\badmit(?:ted|ting)?\b", q)
    ):
        return "referral_hospital"
    if has_ref_word and bool(_PATIENT_WORD.search(q)) and any(
        x in q for x in ("hospital", "facility", "health facility")
    ):
        return "referral_hospital"

    if any(
        x in q
        for x in (
            "dose",
            "dosing",
            "dosage",
            "mg/kg",
            "mg per kg",
            "how much medicine",
            "tablet strength",
        )
    ):
        return "dosing"

    return "general"


def chunk_quality_weight(heading: str, text: str) -> float:
    """
    Multiplier for BM25 score (1.0 = keep, lower = deprioritize junk chunks).
    """
    h = (heading or "").strip()
    if len(h) < 2:
        return 0.08
    low = h.lower()
    if low in _JUNK_HEADINGS:
        return 0.1
    # Very short headings that are often table fragments
    if len(h) <= 3 and sum(1 for c in h if c.isalpha()) < 3:
        return 0.2
    # Mostly punctuation / numbers, almost no letters
    letters = sum(1 for c in h if c.isalpha())
    if letters == 0 and len(h) <= 6:
        return 0.15
    if letters <= 2 and len(h) <= 5:
        return 0.25
    # Long clinical headings are fine
    if len(h) >= 28:
        return 1.05
    return 1.0


def retrieve_top_chunk_indices(
    bm25: Any,
    chunks: List[Dict[str, Any]],
    query_tokens: List[str],
    *,
    k: int = 5,
    pool_min: int = 24,
    pool_factor: int = 6,
) -> List[int]:
    """
    BM25 over a wider pool, then re-rank by ``score * chunk_quality_weight(heading)``.
    Falls back to plain BM25 order if fewer than ``k`` unique indices are found.
    """
    n = len(chunks)
    if n == 0:
        return []
    scores = bm25.get_scores(query_tokens)
    scores = np.asarray(scores, dtype=np.float64)
    pool = min(max(k * pool_factor, pool_min), n)
    order = np.argsort(scores)[::-1][:pool]

    adjusted: List[tuple[int, float]] = []
    for idx in order:
        idx = int(idx)
        ch = chunks[idx]
        w = chunk_quality_weight(ch.get("heading", ""), ch.get("text", ""))
        adjusted.append((idx, float(scores[idx]) * w))
    adjusted.sort(key=lambda x: -x[1])

    out: List[int] = []
    seen: set[int] = set()
    for idx, _ in adjusted:
        if len(out) >= k:
            break
        if idx not in seen:
            out.append(idx)
            seen.add(idx)

    if len(out) < k:
        full_order = np.argsort(scores)[::-1]
        for idx in full_order:
            idx = int(idx)
            if idx not in seen and len(out) < k:
                out.append(idx)
                seen.add(idx)

    return out
