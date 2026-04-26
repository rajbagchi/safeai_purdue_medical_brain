"""
Output layer: VHT-oriented formatting after retrieval + guardrail.

Pipeline order: extraction → validation → chunking → indexing → guardrail → response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .config import MedicalSource, TriageLevel
from .retrieval import infer_query_intent


def severe_malaria_care_context(
    query: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    True when the question or retrieved excerpts indicate severe / cerebral malaria
    (hospital-level care), so VHT text should not use generic 'manage at home' steps.
    """
    q = (query or "").lower()
    # Uncomplicated-only questions: do not infer "severe" from incidental passages
    # deeper in the retrieval list (those often mention severe malaria for contrast).
    if "uncomplicated" in q and "severe" not in q and "cerebral" not in q:
        return False
    if any(p in q for p in ("cerebral malaria", "severe malaria")):
        return True
    if "cerebral" in q and "malaria" in q:
        return True
    if "malaria" in q and (
        "severe" in q or "cerebral" in q or "icu" in q or "unconscious" in q or "coma" in q
    ):
        return True
    if not chunks:
        return False
    # HIV / coinfection questions often retrieve passages that mention "severe"
    # in a different sense; do not escalate triage from chunk text alone.
    if any(x in q for x in ("hiv", "aids", "coinfection", "co-infection", "antiretroviral")):
        return False
    maternal_topic = any(
        x in q
        for x in (
            "pregnant",
            "pregnancy",
            "antenatal",
            "iptp",
            "trimester",
            "first trimester",
            "second trimester",
            "third trimester",
        )
    )
    # Maternal / pregnancy malaria: guideline chunks often say "uncomplicated and severe malaria"
    # in scope lines — skip chunk-only escalation unless the question names severe/cerebral care.
    if maternal_topic and not any(
        x in q for x in ("severe", "cerebral", "danger sign", "icu", "unconscious", "coma")
    ):
        return False
    parts: List[str] = []
    for c in chunks[:6]:
        parts.append(str(c.get("heading", "")))
        parts.append((c.get("text") or "")[:900])
    blob = " ".join(parts).lower()
    # Require explicit severe-malaria wording; strip guideline boilerplate that lists both types.
    if "cerebral malaria" in blob:
        return True
    bl = blob.replace("uncomplicated and severe malaria", "uncomplicated and other malaria")
    if re.search(r"\bsevere\s+malaria\b", bl):
        return True
    return False


_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "how", "does", "this", "that", "with",
    "from", "have", "been", "there", "your", "into", "about", "than", "then", "some",
    "will", "would", "could", "should", "must", "only", "also", "very", "such",
    "each", "other", "these", "those", "them", "they", "their",
})

_JUNK_HEADING_RE = re.compile(
    r"^(references?|untitled|publication|inconsistency|indirectness|imprecision|"
    r"study\s+quality|risk\s+of\s+bias)\b",
    re.I,
)

_BULLET_LINE_RE = re.compile(
    r"^\s*(?:[-*•\u2022\u2610\u25cf]|\d{1,2}[\).])\s*(.+?)\s*$",
    re.MULTILINE,
)

_ACTION_VERB_RE = re.compile(
    r"^\s*(Give|Check|Refer|Ensure|Apply|Administer|Monitor|Observe|Weigh|Take|Start|"
    r"Stop|Continue|Avoid|Use|Treat|Provide|Prepare|Offer|Consider|Maintain|Record|"
    r"Re-?assess|Do\s+not|Do\s+NOT)\b",
    re.I,
)


def _query_terms(query: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[a-zA-Z]{4,}", query or "")
        if t.lower() not in _STOPWORDS
    }


def _chunk_heading_usable(heading: str) -> bool:
    h = (heading or "").strip()
    if len(h) < 3:
        return False
    if _JUNK_HEADING_RE.match(h):
        return False
    hl = h.lower()
    if hl == "references" or hl.startswith("references "):
        return False
    return True


def _line_is_reference_noise(line: str) -> bool:
    low = line.lower()
    if "cochrane database" in low or "pubmed journal" in low:
        return True
    if re.search(r"\bCD\d{6}\b", line):
        return True
    if "publication bias" in low or "inconsistency: no serious" in low:
        return True
    if re.match(r"^\s*\d{2,4}\.\s+", line) and any(
        x in low for x in ("cochrane", "pubmed", "doi:", "discussion e")
    ):
        return True
    return False


def _line_is_author_or_institution_line(line: str) -> bool:
    """Contributor lines, CDC/WHO staffer lines, and TOC dot-leaders are not clinical actions."""
    if re.search(r"\bDr\.?\s+[A-Z]", line):
        return True
    if re.search(
        r"Centers for Disease Control|Division of Parasitic|Malaria Branch,",
        line,
        re.I,
    ):
        return True
    if re.search(r"\.{3,}\s*\d{2,4}\s*$", line):  # dot-leader to page number
        return True
    return False


def _line_is_toc_or_heading_stub(line: str) -> bool:
    """Strip section-number lines (e.g. 2.1.1.1. Duration…) mistaken for care steps."""
    pref = line[:36].strip()
    if not pref:
        return True
    noisy = sum(1 for c in pref if c.isdigit() or c in ".")
    if noisy / max(len(pref), 1) > 0.28:
        return True
    low = line.lower().strip()
    if re.match(r"^\d", line.strip()) and re.search(
        r"(duration of treatment|overview|introduction|references)\s*$",
        low,
    ):
        return True
    if re.match(r"^\d{1,2}(\.\d{1,4}){2,}\s+\S", line) and not _ACTION_VERB_RE.match(
        line
    ):
        return True
    return False


_CLINICAL_SUBSTANCE_RE = re.compile(
    r"(mg|kg|/kg|dose|tablet|capsule|daily|hour|day|refer|diagnos|treat|artem|"
    r"lumefan|amodiaquine|primaquine|rdt|test|patient|child|hospital|facility|"
    r"inject|oral|adult|infant|uncomplicated|severe|parasit|blood|glucose|"
    r"\bact\b|combination|therapy|first[- ]line|second[- ]line|recommended|"
    r"effective|plasmodium|falciparum|vivax|pyron|piperaquine)",
    re.I,
)


def _line_has_clinical_substance(line: str) -> bool:
    return bool(_CLINICAL_SUBSTANCE_RE.search(line))


def _line_clinical_enough_for_malaria(s: str) -> bool:
    """Allow prose bullets that name malaria care without dosing tokens."""
    low = s.lower()
    if "malaria" not in low:
        return False
    return any(
        k in low
        for k in (
            "treat",
            "therap",
            "dose",
            "tablet",
            "act",
            "artem",
            "diagnos",
            "test",
            "combination",
            "first-line",
            "first line",
            "uncomplicated",
            "severe",
        )
    )


def _extract_action_bullets_from_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[str]:
    """
    Pull short clinical lines from retrieved narrative chunks so GREEN VHT actions
    vary with the evidence, not a single generic block.
    """
    q_terms = _query_terms(query)
    scored: List[Tuple[int, str]] = []
    seen: set[str] = set()

    for c in chunks[:10]:
        if c.get("is_table_only"):
            continue
        if not _chunk_heading_usable(str(c.get("heading", ""))):
            continue
        text = (c.get("text") or "")[:3200]
        candidates: List[str] = []
        for m in _BULLET_LINE_RE.finditer(text):
            candidates.append(m.group(1).strip())
        for line in text.splitlines():
            line = line.strip()
            if _ACTION_VERB_RE.match(line):
                candidates.append(line)
        for raw in candidates:
            s = re.sub(r"\*+", "", raw)
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) < 28 or len(s) > 260:
                continue
            if _line_is_reference_noise(s):
                continue
            if _line_is_author_or_institution_line(s):
                continue
            if _line_is_toc_or_heading_stub(s):
                continue
            if not _ACTION_VERB_RE.match(s):
                if not _line_has_clinical_substance(s) and not _line_clinical_enough_for_malaria(
                    s
                ):
                    continue
            low = s.lower()
            key = low[:96]
            if key in seen:
                continue
            seen.add(key)
            score = sum(1 for t in q_terms if t in low)
            if any(k in low for k in ("malaria", "artem", "lumefan", "primaquine", "dose", "tablet", "treat", "patient", "child", "severe", "uncomplicated")):
                score += 2
            if "guideline" in low or "who" in low:
                score += 1
            scored.append((score, s))

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [s for _, s in scored[:8]]


def _extract_monitor_lines_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """Lines that read like monitoring / follow-up from the same narrative pool."""
    pat = re.compile(
        r"(?i)(monitor|observe|watch\s+for|check\s+(?:every|if|the)|record\s+|"
        r"re-?check|follow-?up|blood\s+glucose|urine\s+output|vital\s+signs|"
        r"coma\s+score|consciousness)",
    )
    out: List[str] = []
    seen: set[str] = set()
    for c in chunks[:10]:
        if c.get("is_table_only"):
            continue
        if not _chunk_heading_usable(str(c.get("heading", ""))):
            continue
        for line in (c.get("text") or "").splitlines():
            line = re.sub(r"\s+", " ", line.strip())
            if len(line) < 30 or len(line) > 260:
                continue
            if not pat.search(line):
                continue
            if _line_is_reference_noise(line):
                continue
            if _line_is_author_or_institution_line(line):
                continue
            if line.lower() in seen:
                continue
            seen.add(line.lower())
            out.append(line)
            if len(out) >= 8:
                return out
    return out


def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    out: List[str] = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if 36 <= len(s) <= 280:
            out.append(s)
    return out


def _chunk_ok_for_mining(c: Dict[str, Any]) -> bool:
    if c.get("is_table_only"):
        return False
    h = str(c.get("heading", "")).strip()
    if not h:
        return True
    return _chunk_heading_usable(h)


def _score_sentence_for_query(sent: str, terms: set[str], q_lower: str) -> int:
    low = sent.lower()
    if _line_is_reference_noise(sent) or _line_is_toc_or_heading_stub(sent):
        return -999
    score = sum(3 for t in terms if t in low)
    for w in q_lower.split():
        if len(w) >= 5 and w in low:
            score += 1
    if _ACTION_VERB_RE.match(sent):
        score += 4
    if any(k in low for k in ("malaria", "treat", "dose", "patient", "child", "refer", "hospital")):
        score += 2
    return score


def _dedupe_preserve_order(items: List[str], max_n: int) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.lower().strip()[:140]
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= max_n:
            break
    return out


def _gather_metadata_field(chunks: List[Dict[str, Any]], field: str) -> List[str]:
    out: List[str] = []
    for c in chunks[:12]:
        cm = c.get("clinical_metadata") or {}
        for v in cm.get(field, []) or []:
            if not isinstance(v, str):
                continue
            s = re.sub(r"\s+", " ", v).strip()
            if len(s) < 8 or len(s) > 220:
                continue
            out.append(s)
    return _dedupe_preserve_order(out, 12)


def _sentences_ranked_for_query(
    query: str,
    chunks: List[Dict[str, Any]],
    *,
    exclude: Optional[set[str]] = None,
) -> List[Tuple[int, str]]:
    terms = _query_terms(query)
    q_lower = (query or "").lower()
    excl = exclude or set()
    scored: List[Tuple[int, str]] = []
    for c in chunks[:14]:
        if not _chunk_ok_for_mining(c):
            continue
        text = (c.get("text") or "")[:5000]
        for s in _split_into_sentences(text):
            key = s.lower()[:100]
            if key in excl:
                continue
            sc = _score_sentence_for_query(s, terms, q_lower)
            if sc < -100:
                continue
            if _line_is_author_or_institution_line(s):
                continue
            scored.append((sc, s))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return scored


def _fill_actions_from_sentences(
    query: str,
    chunks: List[Dict[str, Any]],
    existing: List[str],
    target: int = 5,
) -> List[str]:
    have = list(existing)
    excl = {x.lower()[:100] for x in have}
    for sc, s in _sentences_ranked_for_query(query, chunks):
        if s.lower()[:100] in excl:
            continue
        if sc < 1 and len(have) >= 2:
            break
        have.append(s)
        excl.add(s.lower()[:100])
        if len(have) >= target:
            break
    return _dedupe_preserve_order(have, target)


def _fill_monitoring_from_sentences(
    query: str,
    chunks: List[Dict[str, Any]],
    existing: List[str],
    target: int = 6,
) -> List[str]:
    pat = re.compile(
        r"(?i)(monitor|observe|watch|check\s+every|record\s|follow-?up|"
        r"blood\s+glucose|urine\s+output|vital|consciousness|coma\s+score)",
    )
    have = list(existing)
    excl = {x.lower()[:100] for x in have}
    for sc, s in _sentences_ranked_for_query(query, chunks):
        if not pat.search(s):
            continue
        if s.lower()[:100] in excl:
            continue
        have.append(s)
        excl.add(s.lower()[:100])
        if len(have) >= target:
            break
    return _dedupe_preserve_order(have, target)


def _referral_lines_from_prose(query: str, chunks: List[Dict[str, Any]]) -> List[str]:
    pat = re.compile(
        r"(?i)\b(refer|referral|transfer|transport|admit(?:ted|tance)?|hospital|"
        r"health\s+facility|urgent(?:ly)?|immediately|emergency)\b",
    )
    out: List[str] = []
    for sc, s in _sentences_ranked_for_query(query, chunks):
        if not pat.search(s):
            continue
        if len(s) < 40:
            continue
        out.append(s)
        if len(out) >= 6:
            break
    return _dedupe_preserve_order(out, 5)


def _family_sentence_from_evidence(
    query: str,
    chunks: List[Dict[str, Any]],
    actions: List[str],
) -> str:
    edu = re.compile(
        r"(?i)\b(caregiver|family|mother|parent|explain|understand|"
        r"worry|concern|home|together|support)\b",
    )
    terms = _query_terms(query)
    q_lower = (query or "").lower()
    best_s, best_sc = "", -999
    for c in chunks[:12]:
        if not _chunk_ok_for_mining(c):
            continue
        for s in _split_into_sentences((c.get("text") or "")[:4000]):
            if len(s) < 50 or len(s) > 300:
                continue
            if s.lstrip().startswith("#") or s.lstrip().startswith("|"):
                continue
            if _line_is_author_or_institution_line(s):
                continue
            if _line_is_reference_noise(s):
                continue
            sc = _score_sentence_for_query(s, terms, q_lower)
            if edu.search(s):
                sc += 5
            if sc > best_sc:
                best_sc, best_s = sc, s
    if best_sc >= 2 and best_s:
        return best_s
    if actions:
        return actions[0][:300]
    for sc, s in _sentences_ranked_for_query(query, chunks):
        if sc >= 0:
            return s
    return ""


def build_evidence_grounded_bundle(
    query: str,
    triage: TriageLevel,
    intent: str,
    severe_ctx: bool,
    chunks: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str], str, List[str]]:
    """
    Build VHT lists only from retrieval + query semantics (no canned templates).
    """
    _ = triage, intent, severe_ctx  # reserved for future biasing; triage is in footer elsewhere

    actions = _extract_action_bullets_from_chunks(query, chunks)
    if len(actions) < 3:
        actions = _fill_actions_from_sentences(query, chunks, actions, target=5)

    monitoring = _extract_monitor_lines_from_chunks(chunks)
    if len(monitoring) < 2:
        monitoring = _fill_monitoring_from_sentences(query, chunks, monitoring, target=6)

    referral = _gather_metadata_field(chunks, "referral_criteria")
    referral.extend(_referral_lines_from_prose(query, chunks))
    referral = _dedupe_preserve_order(referral, 6)

    danger = _gather_metadata_field(chunks, "danger_signs")

    family = _family_sentence_from_evidence(query, chunks, actions)

    actions = _dedupe_preserve_order(actions, 6)
    if not actions:
        for _, s in _sentences_ranked_for_query(query, chunks):
            actions.append(s)
            if len(actions) >= 2:
                break
    if not actions:
        for c in chunks[:2]:
            blob = (c.get("text") or "").replace("\n", " ")
            for part in re.split(r"(?<=[.!?])\s+", blob):
                s = re.sub(r"\s+", " ", part).strip()
                if 55 <= len(s) <= 260 and not _line_is_reference_noise(s):
                    actions = [s]
                    break
            if actions:
                break

    return (
        actions,
        _dedupe_preserve_order(monitoring, 7),
        referral,
        family,
        danger,
    )


class ResponseFormat(Enum):
    """Output format types."""

    VHT_QUICK = "vht_quick"
    VHT_STANDARD = "vht_standard"
    CLINICIAN = "clinician"
    REFERRAL = "referral"


@dataclass
class ResponseContent:
    """Structured response (output of the response layer)."""

    query: str
    triage: TriageLevel
    triage_reasons: List[str]
    actions: List[str]
    monitoring: List[str]
    referral_criteria: List[str]
    citations: List[Dict[str, Any]]
    medication_dosage: Optional[Dict[str, Any]] = None
    family_message: Optional[str] = None
    danger_signs: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def to_vht_format(self) -> str:
        return VHTResponseFormatter().format(self, ResponseFormat.VHT_STANDARD)


class VHTResponseFormatter:
    """Formats responses for Village Health Teams."""

    TRANSLATIONS = {
        "lethargic": "very weak",
        "unable to drink": "cannot drink",
        "convulsions": "shaking/fitting",
        "dehydration": "dry mouth, no tears",
        "dyspnea": "difficulty breathing",
        "malaria": "fever with chills",
    }

    def format(self, content: ResponseContent, fmt: ResponseFormat) -> str:
        if fmt == ResponseFormat.VHT_QUICK:
            return self._format_quick(content)
        if fmt == ResponseFormat.REFERRAL:
            return self._format_referral_note(content)
        return self._format_standard(content)

    def _format_standard(self, content: ResponseContent) -> str:
        lines: List[str] = []
        if content.triage == TriageLevel.RED:
            rs = "; ".join(content.triage_reasons[:4]) if content.triage_reasons else ""
            lines.append(
                "**RED triage (from your question and guideline match):**\n\n"
                f"• {rs}" if rs else "**RED triage** — see actions from retrieved excerpts below."
            )
        lines.append(self._quick_summary(content))
        act = self._actions_section(content)
        if act:
            lines.append(act)
        if content.monitoring:
            lines.append(self._monitoring_section(content))
        ds = self._danger_signs_section(content)
        if ds:
            lines.append(ds)
        if content.family_message:
            lines.append(self._family_message_section(content.family_message))
        vr = self._vht_reminder_from_citations(content)
        if vr:
            lines.append(vr)
        if content.referral_criteria:
            lines.append(self._referral_criteria_section(content))
        cs = self._citations_section(content.citations)
        if cs:
            lines.append(cs)
        if content.validation_warnings:
            lines.append(self._warnings_section(content.validation_warnings))
        return "\n\n".join(lines)

    def _quick_summary(self, content: ResponseContent) -> str:
        triage_symbol = {
            TriageLevel.RED: "RED (Immediate Referral Required)",
            TriageLevel.YELLOW: "YELLOW (Urgent Referral - Assess Today)",
            TriageLevel.GREEN: "GREEN (Manage at Community Level)",
        }.get(content.triage, "Unknown")
        reasons = ", ".join(content.triage_reasons[:3]) if content.triage_reasons else ""
        if content.actions:
            raw = content.actions[0]
            key_step = raw[:240] + ("…" if len(raw) > 240 else "")
        elif content.monitoring:
            raw = content.monitoring[0]
            key_step = raw[:240] + ("…" if len(raw) > 240 else "")
        elif content.referral_criteria:
            raw = content.referral_criteria[0]
            key_step = raw[:240] + ("…" if len(raw) > 240 else "")
        else:
            key_step = reasons
        return (
            f"**QUICK SUMMARY: {triage_symbol}**\n\n"
            f"• Triage note: {reasons}\n"
            f"• Highest-ranked excerpt step: {key_step}"
        )

    def _actions_section(self, content: ResponseContent) -> str:
        if not content.actions:
            return ""
        lines = ["**WHAT TO DO (step by step):**", ""]
        for i, action in enumerate(content.actions[:5], 1):
            lines.append(f"**Step {i}:** {action}")
        return "\n".join(lines)

    def _monitoring_section(self, content: ResponseContent) -> str:
        lines = ["**MONITORING:**", ""]
        for item in content.monitoring[:6]:
            lines.append(f"• {item}")
        return "\n".join(lines)

    def _referral_criteria_section(self, content: ResponseContent) -> str:
        lines = ["**WHEN TO REFER (from retrieved excerpts / metadata):**", ""]
        for r in content.referral_criteria[:7]:
            lines.append(f"• {r}")
        return "\n".join(lines)

    def _danger_signs_section(self, content: ResponseContent) -> str:
        if not content.danger_signs:
            return ""
        lines = ["**DANGER SIGNS (from retrieved guideline metadata):**", ""]
        for d in content.danger_signs[:8]:
            lines.append(f"• {d}")
        return "\n".join(lines)

    def _family_message_section(self, message: str) -> str:
        return f"**WHAT TO TELL THE FAMILY:**\n\n{message}"

    def _vht_reminder_from_citations(self, content: ResponseContent) -> str:
        pages: List[int] = []
        for c in content.citations:
            p = c.get("page")
            if isinstance(p, int):
                pages.append(p)
        if not pages:
            return ""
        pages_u = sorted(set(pages))[:5]
        pstr = ", ".join(str(p) for p in pages_u)
        return (
            "**SOURCE ANCHOR (this answer):**\n\n"
            f"• Ground community steps in the cited pages ({pstr}) and your national algorithm / supervision."
        )

    def _citations_section(self, citations: List[Dict[str, Any]]) -> str:
        if not citations:
            return ""
        lines = ["**FROM THE GUIDELINES:**", ""]
        for c in citations[:3]:
            src = c.get("source", "Guidelines")
            if isinstance(src, MedicalSource):
                src = src.value
            page = c.get("page", "?")
            section = c.get("section", "")
            if section:
                lines.append(f"• {src}, Page {page}: {section}")
            else:
                lines.append(f"• {src}, Page {page}")
        return "\n".join(lines)

    def _warnings_section(self, warnings: List[str]) -> str:
        lines = ["---", "**GUARDRAIL WARNINGS:**", ""]
        for w in warnings[:3]:
            lines.append(f"• {w}")
        return "\n".join(lines)

    def _format_referral_note(self, content: ResponseContent) -> str:
        lines = [
            "**VHT REFERRAL NOTE**",
            "",
            f"**Triage:** {content.triage.value}",
            f"**Reason:** {', '.join(content.triage_reasons)}",
            "",
        ]
        if content.actions:
            lines.append("**Actions taken:**")
            for a in content.actions[:3]:
                lines.append(f"• {a}")
            lines.append("")
        lines.extend(
            [
                "**Referral completed:** [ ]",
                "**Health worker received:** [ ]",
            ]
        )
        return "\n".join(lines)

    def _format_quick(self, content: ResponseContent) -> str:
        triage_symbol = {
            TriageLevel.RED: "REFER NOW",
            TriageLevel.YELLOW: "REFER TODAY",
            TriageLevel.GREEN: "MANAGE AT HOME",
        }.get(content.triage, "?")
        reasons = ", ".join(content.triage_reasons[:2]) if content.triage_reasons else ""
        act0 = (content.actions[0][:180] + "…") if content.actions else reasons
        return f"{triage_symbol}\nReason: {reasons}\nExcerpt step: {act0}"


class ResponseOrchestrator:
    """
    Builds ResponseContent from retrieval chunks + guardrail output + triage inference.
    """

    def __init__(self, default_format: ResponseFormat = ResponseFormat.VHT_STANDARD):
        self.default_format = default_format
        self.formatter = VHTResponseFormatter()

    def create(
        self,
        *,
        query: str,
        triage: TriageLevel,
        triage_reasons: List[str],
        guardrail_output: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        source: MedicalSource,
        dosage_info: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
    ) -> ResponseContent:
        intent = query_intent if query_intent is not None else infer_query_intent(query)
        severe_ctx = severe_malaria_care_context(query, retrieved_chunks)
        actions, monitoring, referral_criteria, family_message, danger_signs = (
            build_evidence_grounded_bundle(
                query, triage, intent, severe_ctx, retrieved_chunks
            )
        )
        citations = self._build_citations(retrieved_chunks, source)
        warnings = list(guardrail_output.get("warnings", []))
        fm = family_message.strip() if family_message else None
        return ResponseContent(
            query=query,
            triage=triage,
            triage_reasons=triage_reasons,
            actions=actions,
            monitoring=monitoring,
            referral_criteria=referral_criteria,
            citations=citations,
            medication_dosage=dosage_info,
            family_message=fm,
            danger_signs=danger_signs,
            validation_warnings=warnings,
            confidence_score=self._calculate_confidence(
                triage, guardrail_output, intent, len(actions)
            ),
        )

    def _build_citations(
        self,
        chunks: List[Dict[str, Any]],
        source: MedicalSource,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for chunk in chunks[:5]:
            page = chunk.get("page")
            if page is None:
                continue
            out.append({
                "source": source,
                "page": page,
                "section": chunk.get("heading", "Clinical guideline"),
            })
        return out

    def _calculate_confidence(
        self,
        triage: TriageLevel,
        guardrail_output: Dict[str, Any],
        intent: str,
        n_actions: int,
    ) -> float:
        base = 0.95 if triage == TriageLevel.RED else 0.88
        if intent in ("referral_hospital", "dosing"):
            base -= 0.03
        if n_actions < 2:
            base -= 0.1
        if guardrail_output.get("warnings"):
            base -= 0.05 * min(len(guardrail_output["warnings"]), 3)
        if not guardrail_output.get("passed", True):
            base -= 0.1
        return max(0.55, min(1.0, base))


def infer_triage_from_query(query: str) -> tuple[TriageLevel, List[str]]:
    """
    Rule-based triage aligned with danger-sign list used in guardrail footer.
    YELLOW: non-immediate but time-sensitive phrasing in the query, or explicit
    hospital-referral questions (so VHT text is not framed as routine home care).
    """
    q = query.lower()
    reasons: List[str] = []
    danger_kw = (
        ("unable to drink", "Unable to drink / cannot drink"),
        ("cannot drink", "Unable to drink / cannot drink"),
        ("convuls", "Convulsions / seizures"),
        ("seizure", "Convulsions / seizures"),
        ("unconscious", "Unconscious or not waking"),
        ("very weak", "Very weak"),
        ("lethargic", "Very weak / lethargic"),
        ("bleeding", "Bleeding"),
    )
    for needle, label in danger_kw:
        if needle in q:
            reasons.append(label)
    if reasons:
        return TriageLevel.RED, list(dict.fromkeys(reasons))

    if infer_query_intent(query) == "referral_hospital":
        return TriageLevel.YELLOW, [
            "Question asks about hospital referral - use national criteria and danger signs"
        ]

    if "malaria" in q and any(
        x in q for x in ("hiv", "aids", "coinfection", "co-infection", "antiretroviral")
    ):
        return TriageLevel.YELLOW, [
            "HIV–malaria co-infection: use integrated HIV/malaria national protocols and supervised care"
        ]

    if severe_malaria_care_context(query, None):
        return TriageLevel.YELLOW, [
            "Severe or cerebral malaria — hospital-level protocols; not routine home management alone"
        ]

    yellow_triggers = (
        ("fever" in q and ("3 day" in q or ">3" in q or "more than 3" in q)),
        ("cough" in q and ("3 week" in q or ">3" in q)),
        ("urgent" in q and "refer" in q),
        ("assess today" in q),
        ("same day" in q and "refer" in q),
    )
    if any(yellow_triggers):
        return TriageLevel.YELLOW, ["Time-sensitive symptoms — assess at facility today"]

    return TriageLevel.GREEN, ["Routine evidence retrieval from national guidelines"]
