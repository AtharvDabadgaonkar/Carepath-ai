"""
ai_analyst.py
=============

The AI Conversion Copilot.

Design rule for this module: **the model interprets, it never calculates.**

Every figure available to the copilot is pre-computed by ``analytics.py`` and
handed over as a sanitised evidence pack. The system prompt forbids inventing
numbers, and :func:`verify_numbers` checks the generated answer against the
evidence afterwards, flagging any figure that cannot be traced back to the
pack. That check runs on both the live model and the offline fallback.

Three execution modes
---------------------
``llm``
    An API key is configured and the provider responded. The answer is the
    model's, grounded in the evidence pack.
``fallback``
    No API key is configured. A deterministic, rule-based analyst answers from
    the same evidence pack. The app is fully usable in this mode - it is the
    default experience for a reviewer who just clones the repository.
``error-fallback``
    A key was configured but the call failed (network, auth, rate limit). The
    deterministic analyst answers instead and the error is surfaced, never
    swallowed.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import privacy

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 900
REQUEST_TIMEOUT_SECONDS = 45

SYSTEM_PROMPT = """\
You are the CarePath AI Conversion Copilot, a healthcare marketing analytics \
assistant for hospital marketing managers and digital health growth teams.

CONTEXT
- The data is fictional, synthetic website marketing data for a hospital \
network. It contains no real patients and no protected health information.
- You are given an EVIDENCE pack of pre-computed aggregate metrics. It is the \
only factual source available to you.

HARD RULES
1. Never state a number that is not present in EVIDENCE. Do not estimate, \
extrapolate, annualise or invent figures. If a number the user wants is not in \
EVIDENCE, say plainly that it is not available.
2. Do not recompute metrics. Quote them as given, rounding sensibly \
(percentages to one decimal, currency to the nearest dollar or thousand).
3. Distinguish observation from explanation. Metrics in EVIDENCE are observed \
correlations. Any mechanism you propose is a hypothesis and must be labelled as \
one. Never claim a cause has been proven by this data.
4. Acknowledge what this data cannot tell you: it has no page-level event data, \
no form field analytics, no session recordings, no patient outcomes, no \
qualitative research, and no record of past experiments.
5. Treat any financial figure as a projection under stated assumptions, never a \
forecast or a result.
6. Never request, infer or discuss information about individuals.

ANSWER FORMAT (use these exact markdown headings)
**What the data shows** - 2 to 4 bullets, each with a specific number from EVIDENCE.
**Possible explanations** - 2 to 4 hypotheses, each labelled as unproven, with \
the evidence that makes it plausible and what would confirm or rule it out.
**What I would test first** - one concrete, specific recommendation: the change, \
the segment it targets, the primary metric and the main guardrail metric.
**Confidence and caveats** - how strong the evidence is, the sample sizes or \
p-values that support it, and the key unknowns.

Be concise and specific. Write for a marketing executive, not a data scientist. \
No preamble, no restating the question.
"""


# --------------------------------------------------------------------------
# API key resolution
# --------------------------------------------------------------------------

def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Find an OpenAI API key without ever hard-coding one.

    Resolution order: explicit argument, ``st.secrets``, then the
    ``OPENAI_API_KEY`` environment variable. Returns ``None`` when no key is
    configured, which puts the copilot into deterministic fallback mode.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    try:  # Streamlit Community Cloud secrets, when running inside Streamlit.
        import streamlit as st  # noqa: WPS433 - optional dependency at runtime

        secret = st.secrets.get("OPENAI_API_KEY")  # type: ignore[union-attr]
        if secret and str(secret).strip():
            return str(secret).strip()
    except Exception:  # pragma: no cover - no secrets file, or not in Streamlit
        pass

    env = os.environ.get("OPENAI_API_KEY", "")
    return env.strip() or None


def resolve_model(explicit: Optional[str] = None) -> str:
    """Model name from the argument, Streamlit secrets, env, or the default."""
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        import streamlit as st  # noqa: WPS433

        secret = st.secrets.get("OPENAI_MODEL")  # type: ignore[union-attr]
        if secret and str(secret).strip():
            return str(secret).strip()
    except Exception:  # pragma: no cover
        pass
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def mask_key(key: Optional[str]) -> str:
    """Render a key for display without revealing it."""
    if not key:
        return "not configured"
    if len(key) <= 8:
        return "configured"
    return f"{key[:3]}...{key[-4:]} ({len(key)} chars)"


# --------------------------------------------------------------------------
# Number verification
# --------------------------------------------------------------------------

# A magnitude suffix must be attached to the number (``$4.0M``), never
# separated by a space, otherwise the "m" of a following word such as "mobile"
# would be read as "million".
_NUMBER_RE = re.compile(
    r"(?P<currency>\$)?\s?(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>%|\s?percentage points\b|\s?pp\b|k\b|K\b|M\b|m\b|bn\b|B\b)?"
)

_SUFFIX_MULTIPLIERS = {
    "k": 1_000.0,
    "K": 1_000.0,
    "M": 1_000_000.0,
    "m": 1_000_000.0,
    "bn": 1_000_000_000.0,
    "B": 1_000_000_000.0,
}

# Small integers, years and ordinals are not treated as factual claims.
_IGNORE_EXACT = {float(i) for i in range(0, 13)} | {
    2023.0, 2024.0, 2025.0, 2026.0, 100.0, 95.0, 80.0, 90.0, 50.0,
}


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    """Pull numeric claims out of free text, normalising suffixes."""
    found: List[Tuple[str, float]] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).strip()
        try:
            value = float(match.group("value").replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees a number
            continue
        suffix = (match.group("suffix") or "").strip()
        if suffix in _SUFFIX_MULTIPLIERS:
            value *= _SUFFIX_MULTIPLIERS[suffix]
        found.append((raw, value))
    return found


def _flatten_numeric(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _flatten_numeric(value, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten_numeric(item, out)


def evidence_value_set(evidence: Dict[str, Any]) -> List[float]:
    """Every numeric value in the evidence pack, plus common renderings.

    A rate of ``0.4404`` can legitimately be written as ``44.0%``, so every
    value in ``[-1, 1]`` also contributes its percentage form. Larger values do
    **not**, because scaling a five-figure count by 100 would scatter spurious
    candidates across the millions and let fabricated figures slip through.
    Negated forms are always included, since a gap may be quoted either way
    round.
    """
    raw: List[float] = []
    _flatten_numeric(evidence, raw)
    expanded: List[float] = []
    for value in raw:
        expanded.append(value)
        expanded.append(-value)
        if abs(value) <= 1.0:
            expanded.append(value * 100.0)
            expanded.append(-value * 100.0)
    return expanded


def verify_numbers(
    answer: str,
    evidence: Dict[str, Any],
    question: str = "",
    relative_tolerance: float = 0.02,
) -> List[str]:
    """Return the numeric tokens in ``answer`` that the evidence cannot support.

    A token matches if some evidence value is within ``relative_tolerance``
    (default 2%, which absorbs sensible rounding such as ``$4.0M`` for
    ``4,015,534``). Numbers that also appear in the user's own question are
    skipped, as are small integers and year-like values.
    """
    candidates = evidence_value_set(evidence)
    question_values = {round(v, 6) for _, v in extract_numbers(question)}
    unsupported: List[str] = []

    for raw, value in extract_numbers(answer):
        if value in _IGNORE_EXACT or round(value, 6) in question_values:
            continue
        matched = False
        for candidate in candidates:
            if candidate == 0.0:
                if abs(value) < 1e-9:
                    matched = True
                    break
                continue
            if abs(value - candidate) <= abs(candidate) * relative_tolerance:
                matched = True
                break
        if not matched:
            unsupported.append(raw)

    # De-duplicate while preserving order.
    seen: set = set()
    result = []
    for token in unsupported:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


# --------------------------------------------------------------------------
# Response container
# --------------------------------------------------------------------------

@dataclass
class AnalystResponse:
    """A single copilot answer, with everything needed to audit it."""

    question: str
    answer: str
    mode: str  # "llm" | "fallback" | "error-fallback" | "blocked"
    model: Optional[str] = None
    latency_seconds: float = 0.0
    unsupported_numbers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    evidence_sent: Dict[str, Any] = field(default_factory=dict)
    removed_fields: List[str] = field(default_factory=list)
    prompt_preview: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return not self.unsupported_numbers

    @property
    def mode_label(self) -> str:
        return {
            "llm": "Live model",
            "fallback": "Deterministic fallback",
            "error-fallback": "Deterministic fallback (model call failed)",
            "blocked": "Blocked by privacy screening",
        }.get(self.mode, self.mode)


# --------------------------------------------------------------------------
# Formatting helpers shared by the fallback analyst
# --------------------------------------------------------------------------

def fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def fmt_money(value: Any, decimals: int = 0) -> str:
    try:
        return f"${float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"


def _records(evidence: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = evidence.get(key) or []
    return [r for r in value if isinstance(r, dict)]


def _find(records: Sequence[Dict[str, Any]], field_name: str, value: str):
    for record in records:
        if str(record.get(field_name)) == value:
            return record
    return None


def _worst_by(records: Sequence[Dict[str, Any]], metric: str, minimum_sessions: int = 500):
    pool = [r for r in records if float(r.get("sessions", 0)) >= minimum_sessions]
    pool = pool or list(records)
    if not pool:
        return None
    return min(pool, key=lambda r: float(r.get(metric, 0.0)))


def _best_by(records: Sequence[Dict[str, Any]], metric: str, minimum_sessions: int = 500):
    pool = [r for r in records if float(r.get("sessions", 0)) >= minimum_sessions]
    pool = pool or list(records)
    if not pool:
        return None
    return max(pool, key=lambda r: float(r.get(metric, 0.0)))


# --------------------------------------------------------------------------
# Deterministic fallback analyst
# --------------------------------------------------------------------------

INTENTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("device", ("mobile", "desktop", "device", "phone", "responsive")),
    ("channel", ("channel", "paid", "social", "search", "email", "spend",
                 "budget", "campaign", "cac", "acquisition", "roi", "roas")),
    ("experiment", ("experiment", "test", "a/b", "ab test", "impact",
                    "opportunity", "worth", "upside", "value")),
    ("priority", ("first", "priorit", "optimi", "focus", "next", "should we",
                  "where to start")),
    ("dropoff", ("lose", "losing", "loss", "drop", "leak", "abandon",
                 "where are we", "funnel", "biggest")),
]


def classify_intent(question: str) -> str:
    """Route a question to a deterministic answer template."""
    text = (question or "").lower()
    scores: Dict[str, int] = {}
    for intent, keywords in INTENTS:
        scores[intent] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=lambda k: scores[k]) if scores else "overview"
    return best if scores.get(best, 0) > 0 else "overview"


def _device_answer(evidence: Dict[str, Any]) -> str:
    devices = _records(evidence, "by_device")
    mobile = _find(devices, "device", "Mobile") or {}
    desktop = _find(devices, "device", "Desktop") or {}
    test = evidence.get("mobile_vs_desktop_completion_test", {}) or {}
    opportunities = _records(evidence, "ranked_opportunities")
    mobile_opp = next(
        (o for o in opportunities if str(o.get("segment")) == "Mobile"), None
    )

    lines = ["**What the data shows**"]
    lines.append(
        f"- Mobile sessions complete the booking form at "
        f"{fmt_pct(mobile.get('completion_rate'))} versus "
        f"{fmt_pct(desktop.get('completion_rate'))} on desktop - a gap of "
        f"{fmt_pct(abs(float(test.get('difference', 0.0))))} absolute."
    )
    lines.append(
        f"- The gap is not sampling noise: "
        f"{fmt_int(test.get('n_a'))} mobile booking starts versus "
        f"{fmt_int(test.get('n_b'))} desktop, two-proportion z = "
        f"{test.get('z_statistic')}, p = {test.get('p_value')}."
    )
    lines.append(
        f"- Mobile also starts fewer bookings per provider search "
        f"({fmt_pct(mobile.get('start_rate'))} versus "
        f"{fmt_pct(desktop.get('start_rate'))}), so the problem begins before "
        f"the form, not only inside it."
    )
    lines.append(
        f"- End to end, mobile converts {fmt_pct(mobile.get('session_to_booking_rate'))} "
        f"of sessions into completed bookings against "
        f"{fmt_pct(desktop.get('session_to_booking_rate'))} on desktop, on "
        f"{fmt_int(mobile.get('sessions'))} mobile sessions."
    )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- Form friction on small screens: a long single-page booking form with "
        "date pickers and insurance fields is harder to complete on a phone. "
        "Field-level drop-off instrumentation would confirm or rule this out."
    )
    lines.append(
        "- Intent mix, not usability: mobile traffic skews toward channels with "
        "weaker intent, so a share of the gap may be *who* arrives rather than "
        "*what* they experience. Comparing mobile and desktop within a single "
        "channel would separate the two."
    )
    lines.append(
        "- Interruption and context: phone sessions are more likely to be "
        "interrupted mid-form. A resumable or save-and-return flow would test this."
    )
    lines.append(
        "- Technical performance: slower load or layout shift on mobile. "
        "Core Web Vitals by device would test this; the dataset has no page "
        "performance data."
    )

    lines.append("")
    lines.append("**What I would test first**")
    if mobile_opp:
        lines.append(
            f"- Rebuild the mobile booking form as a short multi-step flow with a "
            f"persistent progress indicator, targeted at mobile sessions that reach "
            f"the booking start step. Primary metric: booking-start to completion "
            f"rate on mobile (baseline {fmt_pct(mobile.get('completion_rate'))}). "
            f"Guardrail: attended rate and no-show rate, so faster completion does "
            f"not simply produce appointments nobody keeps."
        )
        lines.append(
            f"- At full parity with desktop this gap is worth about "
            f"{fmt_int(mobile_opp.get('recoverable_attended_appointments'))} "
            f"additional attended appointments and "
            f"{fmt_money(mobile_opp.get('projected_margin_opportunity'))} of "
            f"contribution margin over the same {evidence.get('data_scope', {}).get('days_in_window')}-day "
            f"window. That is a projected ceiling under a parity assumption, not a forecast."
        )
    else:
        lines.append(
            "- Rebuild the mobile booking form as a short multi-step flow and "
            "measure booking-start to completion rate on mobile."
        )

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- Confidence that a real mobile gap exists: high. The sample is large "
        "and the difference is far outside the confidence interval "
        f"[{test.get('ci_low')}, {test.get('ci_high')}]."
    )
    lines.append(
        "- Confidence in the *cause*: low. This is observational, aggregated "
        "data with no page-level events, no form analytics and no session "
        "recordings. Device is correlated with channel, time of day and intent, "
        "none of which are controlled for here."
    )
    lines.append(
        "- Full parity with desktop is almost certainly unreachable from a "
        "single change. Treat the parity figure as an upper bound and plan "
        "against a partial capture scenario in the Experiment Studio."
    )
    return "\n".join(lines)


def _dropoff_answer(evidence: Dict[str, Any]) -> str:
    leak = evidence.get("largest_drop_off", {}) or {}
    funnel = _records(evidence, "funnel")
    kpi = evidence.get("kpi_summary", {}) or {}
    opportunities = _records(evidence, "ranked_opportunities")

    lines = ["**What the data shows**"]
    lines.append(
        f"- The largest single loss of volume is **{leak.get('step_label')}**: "
        f"{fmt_int(leak.get('lost'))} visitors lost, "
        f"{fmt_pct(leak.get('drop_off_rate'))} of the stage above."
    )
    for row in funnel[1:]:
        lines.append(
            f"- {row.get('stage_label')}: {fmt_int(row.get('count'))} "
            f"({fmt_pct(row.get('step_conversion'))} of the previous stage, "
            f"{fmt_int(row.get('drop_off'))} lost)."
        )
    lines.append(
        f"- Overall, {fmt_int(kpi.get('sessions'))} sessions produced "
        f"{fmt_int(kpi.get('bookings_completed'))} completed bookings "
        f"({fmt_pct(kpi.get('session_to_booking_rate'))}) and "
        f"{fmt_int(kpi.get('appointments_attended'))} attended appointments."
    )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- Biggest absolute loss is not the same as biggest addressable loss. "
        "Most sessions that never reach a provider search were probably never "
        "booking-intent sessions at all, so a large share of that drop is not "
        "recoverable."
    )
    lines.append(
        "- The steps further down the funnel are where intent is already "
        "demonstrated, so a point of conversion recovered there is worth more "
        "than a point recovered at the top."
    )
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- Ranked by gap against the best-performing comparable segment, the "
            f"largest *addressable* loss is {top.get('segment')} at "
            f"'{top.get('step')}': {fmt_pct(top.get('segment_rate'))} versus "
            f"{fmt_pct(top.get('benchmark_rate'))} for "
            f"{top.get('benchmark_segment')}."
        )

    lines.append("")
    lines.append("**What I would test first**")
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- Target {top.get('segment')} at the '{top.get('step')}' step. "
            f"Closing that gap fully would be worth roughly "
            f"{fmt_int(top.get('recoverable_attended_appointments'))} attended "
            f"appointments and "
            f"{fmt_money(top.get('projected_margin_opportunity'))} in projected "
            f"contribution margin over this window. Primary metric: that step's "
            f"conversion rate for that segment. Guardrail: attended rate, so a "
            f"conversion lift is not bought with no-shows."
        )
    else:
        lines.append("- No segment gap large enough to size under the current filters.")

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- The funnel counts themselves are exact; they are sums over "
        f"{fmt_int(evidence.get('data_scope', {}).get('rows_aggregated'))} aggregated rows."
    )
    lines.append(
        "- Opportunity figures are projections that assume a lagging segment can "
        "reach a peer's observed rate. That assumption is untested."
    )
    lines.append(
        "- This data cannot show *where inside a step* people leave - there are "
        "no page views, no form field events and no error logs in scope."
    )
    return "\n".join(lines)


def _channel_answer(evidence: Dict[str, Any]) -> str:
    channels = _records(evidence, "by_channel")
    paid = [c for c in channels if float(c.get("marketing_spend", 0)) > 0]
    kpi = evidence.get("kpi_summary", {}) or {}

    worst_cpa = max(paid, key=lambda c: float(c.get("cost_per_attended", 0.0))) if paid else None
    worst_conv = _worst_by(channels, "session_to_booking_rate")
    best_conv = _best_by(channels, "session_to_booking_rate")

    lines = ["**What the data shows**"]
    if worst_cpa:
        lines.append(
            f"- **{worst_cpa.get('channel')}** is the least efficient paid channel: "
            f"{fmt_money(worst_cpa.get('cost_per_attended'), 2)} per attended "
            f"appointment on {fmt_money(worst_cpa.get('marketing_spend'))} of spend, "
            f"returning {float(worst_cpa.get('return_on_spend', 0)):.1f}x in "
            f"contribution margin."
        )
    if worst_conv:
        lines.append(
            f"- {worst_conv.get('channel')} converts "
            f"{fmt_pct(worst_conv.get('session_to_booking_rate'))} of its "
            f"{fmt_int(worst_conv.get('sessions'))} sessions into completed "
            f"bookings."
        )
    if best_conv:
        lines.append(
            f"- The strongest converter is {best_conv.get('channel')} at "
            f"{fmt_pct(best_conv.get('session_to_booking_rate'))} on "
            f"{fmt_int(best_conv.get('sessions'))} sessions."
        )
    lines.append(
        f"- Blended, {fmt_money(kpi.get('marketing_spend'))} of spend sits behind "
        f"{fmt_int(kpi.get('appointments_attended'))} attended appointments, "
        f"{fmt_money(kpi.get('cost_per_attended'), 2)} each."
    )
    if worst_conv:
        lines.append(
            f"- {worst_conv.get('channel')} loses most of its traffic early: only "
            f"{fmt_pct(worst_conv.get('search_rate'))} of its sessions reach a "
            f"provider search and {fmt_pct(worst_conv.get('start_rate'))} of those "
            f"start a booking."
        )

    period = evidence.get("period_comparison", {}) or {}
    if period.get("available"):
        cpa = (period.get("metrics", {}) or {}).get("cost_per_attended", {})
        if cpa:
            lines.append(
                f"- Cost per attended appointment moved from "
                f"{fmt_money(cpa.get('prior'), 2)} in the first half of the window "
                f"to {fmt_money(cpa.get('recent'), 2)} in the second "
                f"({fmt_pct(cpa.get('relative_change'))})."
            )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- Audience intent, not creative quality: interruption-based channels "
        "reach people who were not looking for care, so they browse and leave. "
        "The early-funnel drop in the weakest channel is consistent with this."
    )
    lines.append(
        "- Device composition: the weakest channel also skews most heavily "
        "mobile, and mobile converts far worse at the booking step. Part of the "
        "channel gap may be the device gap wearing a different hat. A "
        "channel-by-device split would separate them."
    )
    lines.append(
        "- Landing experience mismatch: traffic may land on general pages rather "
        "than a bookable provider page. The dataset has no landing-page data, so "
        "this is untestable here."
    )
    lines.append(
        "- Budget expansion diluting quality: if spend scaled recently, the "
        "marginal audience is usually worse than the core audience."
    )

    lines.append("")
    lines.append("**What I would test first**")
    if worst_cpa:
        lines.append(
            f"- Do not cut {worst_cpa.get('channel')} on cost per appointment alone; "
            f"it still returns {float(worst_cpa.get('return_on_spend', 0)):.1f}x. "
            f"Instead run a landing-experience test for that channel: route its "
            f"traffic straight to a bookable provider page. Primary metric: "
            f"session to booking-start rate for that channel "
            f"(baseline {fmt_pct(worst_cpa.get('start_rate'))}). Guardrail: cost "
            f"per attended appointment and total attended volume."
        )

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- Channel totals are exact sums; the efficiency ranking is reliable."
    )
    lines.append(
        f"- Spend is a lower bound: "
        f"{fmt_int((evidence.get('data_quality', {}) or {}).get('spend_rows_missing'))} "
        f"rows have unknown spend and are excluded from cost metrics."
    )
    lines.append(
        "- Channels are not randomly assigned. Differences between them conflate "
        "audience, device, intent and creative, so none of this is causal."
    )
    lines.append(
        "- Attribution is last-touch by construction in this dataset; assisted "
        "contribution from upper-funnel channels is invisible here."
    )
    return "\n".join(lines)


def _priority_answer(evidence: Dict[str, Any]) -> str:
    opportunities = _records(evidence, "ranked_opportunities")
    kpi = evidence.get("kpi_summary", {}) or {}
    days = (evidence.get("data_scope", {}) or {}).get("days_in_window")

    lines = ["**What the data shows**"]
    if not opportunities:
        lines.append("- No parity gaps are large enough to rank under the current filters.")
    for i, opp in enumerate(opportunities[:4], start=1):
        lines.append(
            f"- #{i} {opp.get('segment')} at '{opp.get('step')}': "
            f"{fmt_pct(opp.get('segment_rate'))} versus "
            f"{fmt_pct(opp.get('benchmark_rate'))} for "
            f"{opp.get('benchmark_segment')} - a projected "
            f"{fmt_money(opp.get('projected_margin_opportunity'))} in contribution "
            f"margin and {fmt_int(opp.get('recoverable_attended_appointments'))} "
            f"attended appointments over {days} days at full parity."
        )
    lines.append(
        f"- For scale: the whole window produced "
        f"{fmt_int(kpi.get('appointments_attended'))} attended appointments and "
        f"{fmt_money(kpi.get('contribution_margin'))} of estimated contribution margin."
    )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- The top-ranked gaps cluster around a single mechanism rather than "
        "several independent ones, which is why they should not simply be added "
        "together - fixing one will move the others."
    )
    lines.append(
        "- Gaps measured against a peer segment assume the peer is a fair "
        "benchmark. If the segments differ in intent, the benchmark is too "
        "generous and the opportunity is overstated."
    )

    lines.append("")
    lines.append("**What I would test first**")
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- Start with {top.get('segment')} at '{top.get('step')}'. It has the "
            f"largest projected value, the clearest single owner, and it sits "
            f"late in the funnel where intent is already demonstrated, so a "
            f"point of lift is worth more than a point at the top."
        )
        lines.append(
            "- Sequence the rest behind it rather than running them in parallel: "
            "the opportunities overlap, so simultaneous tests will contaminate "
            "each other's measurement."
        )
    lines.append(
        "- Open the Experiment Studio to turn this into a powered A/B test with "
        "a sample-size requirement, guardrails and an approval step."
    )

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- The ranking is deterministic and reproducible, but it is a ranking of "
        "*measured gaps*, not of *expected results*. It says nothing about how "
        "hard each gap is to close."
    )
    lines.append(
        "- Every value shown is a projection at full parity, which is an upper "
        "bound. Use the capture-rate control to plan against a realistic share."
    )
    lines.append(
        "- Implementation cost and engineering effort are not in this dataset "
        "and are not reflected in the ranking."
    )
    return "\n".join(lines)


def _experiment_answer(evidence: Dict[str, Any]) -> str:
    opportunities = _records(evidence, "ranked_opportunities")
    devices = _records(evidence, "by_device")
    mobile = _find(devices, "device", "Mobile") or {}
    days = (evidence.get("data_scope", {}) or {}).get("days_in_window")

    lines = ["**What the data shows**"]
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- The largest measured opportunity is {top.get('segment')} at "
            f"'{top.get('step')}', worth a projected "
            f"{fmt_money(top.get('projected_margin_opportunity'))} and "
            f"{fmt_int(top.get('recoverable_attended_appointments'))} attended "
            f"appointments over {days} days at full parity with "
            f"{top.get('benchmark_segment')}."
        )
    for opp in opportunities[1:3]:
        lines.append(
            f"- Next largest: {opp.get('segment')} at '{opp.get('step')}', "
            f"{fmt_money(opp.get('projected_margin_opportunity'))} projected."
        )
    if mobile:
        lines.append(
            f"- The mobile booking step has "
            f"{fmt_int(mobile.get('booking_starts'))} booking starts in the "
            f"window, which is ample traffic to power a test quickly."
        )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- A measurable gap exists because the experience differs, in which case "
        "an experience change should move it. This is the assumption the "
        "experiment exists to test."
    )
    lines.append(
        "- Alternatively the gap reflects audience composition, in which case an "
        "experience change will move nothing - which is exactly why this should "
        "be a controlled test rather than a rollout."
    )

    lines.append("")
    lines.append("**What I would test first**")
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- Hypothesis: if we reduce friction for {top.get('segment')} at the "
            f"'{top.get('step')}' step, that step's conversion rate will rise from "
            f"{fmt_pct(top.get('segment_rate'))} toward "
            f"{fmt_pct(top.get('benchmark_rate'))}."
        )
        lines.append(
            "- Design: 50/50 randomised split, primary metric is that step's "
            "conversion rate, guardrails are attended rate, no-show rate and cost "
            "per attended appointment. Run to a pre-computed sample size, not to "
            "significance."
        )
        lines.append(
            "- Size the financial case in the Experiment Studio with an explicit "
            "capture-rate assumption rather than assuming full parity."
        )

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- Any financial number attached to an experiment before it runs is a "
        "projection built on an assumed lift. It is a planning input, not a result."
    )
    lines.append(
        "- The simulator in the Experiment Studio generates synthetic outcomes "
        "from an assumed true lift. It demonstrates how the readout would work; "
        "it is not evidence that the lift exists."
    )
    lines.append(
        "- Running several of these at once on overlapping traffic will make all "
        "of their readouts unreliable."
    )
    return "\n".join(lines)


def _overview_answer(evidence: Dict[str, Any]) -> str:
    kpi = evidence.get("kpi_summary", {}) or {}
    leak = evidence.get("largest_drop_off", {}) or {}
    opportunities = _records(evidence, "ranked_opportunities")
    scope = evidence.get("data_scope", {}) or {}

    lines = ["**What the data shows**"]
    lines.append(
        f"- Over {scope.get('days_in_window')} days: "
        f"{fmt_int(kpi.get('sessions'))} sessions, "
        f"{fmt_int(kpi.get('bookings_completed'))} completed bookings "
        f"({fmt_pct(kpi.get('session_to_booking_rate'))}) and "
        f"{fmt_int(kpi.get('appointments_attended'))} attended appointments."
    )
    lines.append(
        f"- {fmt_money(kpi.get('marketing_spend'))} of marketing spend, "
        f"{fmt_money(kpi.get('cost_per_attended'), 2)} per attended appointment, "
        f"{fmt_money(kpi.get('contribution_margin'))} of estimated contribution margin."
    )
    lines.append(
        f"- Largest volume loss: {leak.get('step_label')}, "
        f"{fmt_int(leak.get('lost'))} visitors."
    )
    lines.append(
        f"- {fmt_pct(kpi.get('no_show_rate'))} of completed bookings did not "
        f"result in an attended appointment."
    )

    lines.append("")
    lines.append("**Possible explanations** (hypotheses, none proven by this data)")
    lines.append(
        "- Blended rates hide segment behaviour. The segment views are where the "
        "actionable differences live."
    )
    lines.append(
        "- A high top-of-funnel loss is normal for a hospital website, where most "
        "sessions are informational rather than booking-intent."
    )

    lines.append("")
    lines.append("**What I would test first**")
    if opportunities:
        top = opportunities[0]
        lines.append(
            f"- The ranked opportunity list puts {top.get('segment')} at "
            f"'{top.get('step')}' first, at a projected "
            f"{fmt_money(top.get('projected_margin_opportunity'))}. Ask a more "
            f"specific question - about mobile, about channels, or about what to "
            f"prioritise - for a fuller answer."
        )
    else:
        lines.append(
            "- Widen the filters: the current selection has no segment gap large "
            "enough to rank."
        )

    lines.append("")
    lines.append("**Confidence and caveats**")
    lines.append(
        "- This is a summary of aggregated, synthetic data. Nothing here is "
        "causal and no figure describes a real organisation."
    )
    lines.append(
        "- Financial figures use assumed contribution margins per attended "
        "appointment, adjustable in the Experiment Studio."
    )
    return "\n".join(lines)


_INTENT_HANDLERS = {
    "device": _device_answer,
    "dropoff": _dropoff_answer,
    "channel": _channel_answer,
    "priority": _priority_answer,
    "experiment": _experiment_answer,
    "overview": _overview_answer,
}


def deterministic_answer(question: str, evidence: Dict[str, Any]) -> str:
    """Answer a question from the evidence pack with no model call at all."""
    intent = classify_intent(question)
    handler = _INTENT_HANDLERS.get(intent, _overview_answer)
    return handler(evidence)


# --------------------------------------------------------------------------
# The copilot
# --------------------------------------------------------------------------

def build_messages(question: str, evidence: Dict[str, Any]) -> List[Dict[str, str]]:
    """Assemble the exact chat messages that would be sent to the provider."""
    import json

    user_content = (
        "EVIDENCE (aggregated synthetic marketing metrics - the only facts "
        "available to you):\n"
        f"{json.dumps(evidence, indent=2, default=str)}\n\n"
        f"QUESTION: {question.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class AIAnalyst:
    """Facade over the language model with a deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        force_fallback: bool = False,
    ) -> None:
        self.api_key = resolve_api_key(api_key)
        self.model = resolve_model(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.force_fallback = bool(force_fallback)

    # -- state -------------------------------------------------------------

    @property
    def api_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def available(self) -> bool:
        """Whether a live model call will actually be attempted."""
        return self.api_configured and not self.force_fallback

    def status(self) -> Dict[str, Any]:
        return {
            "api_configured": self.api_configured,
            "key": mask_key(self.api_key),
            "model": self.model if self.api_configured else None,
            "mode": "llm" if self.available else "fallback",
            "forced_fallback": self.force_fallback,
        }

    # -- the one public entry point ---------------------------------------

    def ask(
        self,
        question: str,
        evidence: Dict[str, Any],
        screen: bool = True,
    ) -> AnalystResponse:
        """Answer ``question`` using only ``evidence``.

        The evidence pack is sanitised through the privacy allowlist first, and
        the question is screened. Whatever happens afterwards - live model,
        fallback, or a failed call - the response object records which path was
        taken and which numbers could not be traced back to the evidence.
        """
        question = (question or "").strip()

        if screen:
            screening = privacy.screen_question(question)
            if not screening.allowed:
                return AnalystResponse(
                    question=question,
                    answer=screening.message,
                    mode="blocked",
                    warnings=[f"Blocked: {', '.join(screening.findings)}"],
                )

        sanitized = privacy.sanitize_evidence(evidence)
        payload = sanitized.payload
        warnings: List[str] = []
        if sanitized.removed:
            warnings.append(
                f"{len(sanitized.removed)} field(s) were removed by the privacy "
                f"allowlist before sending: {', '.join(sanitized.removed[:6])}"
            )
        leaked = privacy.assert_payload_is_safe(payload)
        if leaked:
            return AnalystResponse(
                question=question,
                answer=(
                    "The outbound payload failed its final privacy check and was "
                    "not sent. This is a bug - please report it."
                ),
                mode="blocked",
                warnings=[f"Payload scan flagged: {', '.join(leaked)}"],
                evidence_sent=payload,
                removed_fields=sanitized.removed,
            )

        messages = build_messages(question, payload)
        prompt_preview = messages[1]["content"]

        if not self.available:
            started = time.perf_counter()
            answer = deterministic_answer(question, payload)
            mode = "fallback"
            if self.force_fallback and self.api_configured:
                warnings.append(
                    "Fallback mode is switched on, so no request was sent even "
                    "though an API key is configured."
                )
            else:
                warnings.append(
                    "No API key configured. This answer was produced by the "
                    "deterministic rule-based analyst from the same evidence "
                    "pack. Nothing was transmitted."
                )
            return self._finalise(
                question, answer, mode, None, time.perf_counter() - started,
                payload, sanitized.removed, prompt_preview, warnings, {},
            )

        started = time.perf_counter()
        try:
            answer, usage = self._call_model(messages)
            mode = "llm"
            model_used: Optional[str] = self.model
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, never hidden
            answer = deterministic_answer(question, payload)
            mode = "error-fallback"
            model_used = None
            usage = {}
            warnings.append(
                f"The model request failed ({type(exc).__name__}: {exc}). "
                "The deterministic analyst answered instead."
            )

        return self._finalise(
            question, answer, mode, model_used, time.perf_counter() - started,
            payload, sanitized.removed, prompt_preview, warnings, usage,
        )

    # -- internals ---------------------------------------------------------

    def _call_model(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        from openai import OpenAI  # imported lazily so the app starts without it

        client = OpenAI(api_key=self.api_key, timeout=REQUEST_TIMEOUT_SECONDS)
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("The model returned an empty response.")
        usage: Dict[str, Any] = {}
        if getattr(completion, "usage", None):
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        return text, usage

    def _finalise(
        self,
        question: str,
        answer: str,
        mode: str,
        model: Optional[str],
        latency: float,
        payload: Dict[str, Any],
        removed: List[str],
        prompt_preview: str,
        warnings: List[str],
        usage: Dict[str, Any],
    ) -> AnalystResponse:
        unsupported = verify_numbers(answer, payload, question)
        if unsupported:
            warnings.append(
                f"{len(unsupported)} figure(s) in this answer could not be matched "
                f"to the evidence pack: {', '.join(unsupported[:8])}. Treat them "
                "as unverified."
            )
        return AnalystResponse(
            question=question,
            answer=answer,
            mode=mode,
            model=model,
            latency_seconds=round(latency, 3),
            unsupported_numbers=unsupported,
            warnings=warnings,
            evidence_sent=payload,
            removed_fields=removed,
            prompt_preview=prompt_preview,
            usage=usage,
        )


SUGGESTED_QUESTIONS: List[str] = [
    "Why are mobile visitors converting less?",
    "Where are the largest booking losses?",
    "Which marketing channel is underperforming?",
    "What should we optimize first?",
    "Which experiment offers the greatest potential impact?",
]
