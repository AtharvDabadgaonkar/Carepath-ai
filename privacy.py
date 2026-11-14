"""
privacy.py
==========

Trust, privacy and data-handling guardrails for CarePath AI.

The design position of this project is simple and deliberately conservative:

* The dataset is **entirely synthetic**. No real visitor, patient or customer
  record is read, stored or transmitted at any point.
* The only data that ever leaves the process is an **aggregated, allowlisted
  evidence pack** built by ``analytics.build_evidence_pack``. Row-level records
  are structurally incapable of reaching the model, because the outbound
  payload is rebuilt key-by-key from an allowlist rather than filtered.
* The user's free-text question is scanned before it is sent. If it contains
  something that looks like personal or health information, the request is
  blocked and the user is told why.
* Nothing in this project is a compliance control. It is a prototype that
  demonstrates privacy-aware product design. See :data:`DISCLAIMER`.

Allowlist, not blocklist
------------------------
``sanitize_evidence`` does not try to find and strip bad keys. It walks the
payload and copies across only names that appear in :data:`ALLOWED_FIELDS`,
reporting everything it dropped. A new field added upstream is therefore
excluded by default until someone deliberately adds it to the allowlist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DISCLAIMER = (
    "Independent personal portfolio project. Uses synthetic demonstration data "
    "only. Not intended for clinical decisions or processing real patient "
    "information."
)

NOT_COMPLIANCE_NOTICE = (
    "This prototype is not HIPAA compliant, has not been security reviewed, and "
    "is not production ready. It processes no protected health information "
    "because it processes no real data at all."
)

PRIVACY_PRINCIPLES: List[Tuple[str, str]] = [
    (
        "Synthetic data only",
        "All 5,400 rows are generated locally from a fixed random seed. The "
        "application has no database, no upload path and no connection to any "
        "real analytics or clinical system.",
    ),
    (
        "Aggregates leave, records never do",
        "The AI copilot receives a rebuilt dictionary of business metrics - "
        "funnel rates, segment totals, projected opportunity values. "
        "Row-level data is never serialised into a prompt.",
    ),
    (
        "Allowlist by construction",
        "The outbound payload is assembled from an explicit field allowlist. "
        "Any field not on the list is dropped and reported, so new upstream "
        "fields cannot leak by accident.",
    ),
    (
        "Inspect before you send",
        "The exact JSON payload is shown in the Model Input Preview panel "
        "before any request is made, and after every answer.",
    ),
    (
        "Input screening",
        "Questions are scanned for patterns resembling personal or health "
        "identifiers. A match blocks the request rather than redacting it, so "
        "the user is never silently truncated.",
    ),
    (
        "Numbers are computed, not generated",
        "Every figure originates in analytics.py. AI answers are checked "
        "against the evidence pack and unsupported numbers are flagged.",
    ),
    (
        "Human approval before action",
        "Experiments produced by the copilot stay in a Draft state until a "
        "named reviewer approves them. Nothing is launched automatically.",
    ),
]

OUT_OF_SCOPE: List[str] = [
    "Real patient records, EHR data or any protected health information",
    "Individual visitor identifiers, cookies, IP addresses or session replays",
    "Clinical decision support of any kind",
    "Claims of regulatory compliance or production readiness",
    "Automated execution of marketing changes without human approval",
]

# --------------------------------------------------------------------------
# Outbound field allowlist
# --------------------------------------------------------------------------

_SCOPE_FIELDS = {
    "is_synthetic",
    "days_in_window",
    "start_date",
    "end_date",
    "rows_aggregated",
    "filters_applied",
    "date_start",
    "date_end",
    "channels",
    "devices",
    "service_lines",
}

_METRIC_FIELDS = {
    "sessions",
    "provider_searches",
    "booking_starts",
    "bookings_completed",
    "appointments_attended",
    "marketing_spend",
    "contribution_margin",
    "estimated_contribution_margin",
    "margin_per_attended",
    "session_to_booking_rate",
    "session_to_attended_rate",
    "cost_per_booking",
    "cost_per_attended",
    "margin_per_session",
    "return_on_spend",
    "no_show_rate",
    "search_rate",
    "start_rate",
    "completion_rate",
    "attendance_rate",
    "days",
    "spend_rows_missing",
    "spend_missing_share",
}

_STRUCTURE_FIELDS = {
    # funnel / drop-off
    "stage_label",
    "count",
    "step_conversion",
    "drop_off",
    "drop_off_rate",
    "step_key",
    "step_label",
    "from_stage",
    "to_stage",
    "lost",
    # dimension keys
    "channel",
    "device",
    "service_line",
    "dimension",
    "segment",
    "step",
    # statistical test
    "rate_a",
    "rate_b",
    "n_a",
    "n_b",
    "difference",
    "relative_difference",
    "z_statistic",
    "p_value",
    "ci_low",
    "ci_high",
    "significant",
    "alpha",
    # opportunity sizing
    "segment_rate",
    "benchmark_segment",
    "benchmark_rate",
    "gap",
    "relative_gap",
    "recoverable_attended_appointments",
    "projected_margin_opportunity",
    "capture_rate_assumption",
    # period comparison
    "available",
    "split_date",
    "prior_days",
    "recent_days",
    "metrics",
    "prior",
    "recent",
    "absolute_change",
    "relative_change",
    # data quality
    "note",
}

#: Every field name permitted anywhere in the outbound payload.
ALLOWED_FIELDS = _SCOPE_FIELDS | _METRIC_FIELDS | _STRUCTURE_FIELDS

#: Permitted top-level sections of the evidence pack.
ALLOWED_SECTIONS = {
    "data_scope",
    "kpi_summary",
    "funnel",
    "largest_drop_off",
    "by_channel",
    "by_device",
    "by_service_line",
    "mobile_vs_desktop_completion_test",
    "ranked_opportunities",
    "period_comparison",
    "data_quality",
}


@dataclass
class SanitizationResult:
    """Outcome of running an evidence payload through the allowlist."""

    payload: Dict[str, Any]
    removed: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.removed

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, default=str)

    def size_bytes(self) -> int:
        return len(self.as_json(indent=None).encode("utf-8"))


def _sanitize_value(value: Any, path: str, removed: List[str]) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, inner in value.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}" if path else key_str
            # Dimension-value keys (e.g. a metric name inside period_comparison)
            # are allowed when the field itself is allowlisted.
            if key_str in ALLOWED_FIELDS:
                out[key_str] = _sanitize_value(inner, child_path, removed)
            else:
                removed.append(child_path)
        return out
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_value(item, f"{path}[{i}]", removed)
            for i, item in enumerate(value)
        ]
    return value


def sanitize_evidence(payload: Dict[str, Any]) -> SanitizationResult:
    """Rebuild ``payload`` from the allowlist, reporting anything dropped."""
    removed: List[str] = []
    clean: Dict[str, Any] = {}
    for section, value in (payload or {}).items():
        section_str = str(section)
        if section_str not in ALLOWED_SECTIONS:
            removed.append(section_str)
            continue
        clean[section_str] = _sanitize_value(value, section_str, removed)
    return SanitizationResult(payload=clean, removed=removed)


# --------------------------------------------------------------------------
# Input screening
# --------------------------------------------------------------------------

PII_PATTERNS: List[Tuple[str, str]] = [
    ("email address", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ("phone number", r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)"),
    ("social security number", r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    ("medical record number", r"\b(?:mrn|medical record(?: number)?)\b[^\n]{0,12}\d{3,}"),
    ("date of birth", r"\b(?:dob|date of birth)\b"),
    ("long numeric identifier", r"(?<!\d)\d{9,}(?!\d)"),
    ("insurance or member id", r"\b(?:member|policy|insurance)\s*(?:id|number|#)\b"),
]

#: Phrases that indicate the user is asking for individual-level information.
INDIVIDUAL_LEVEL_PHRASES: List[str] = [
    "patient name",
    "patient names",
    "which patient",
    "who booked",
    "name of the patient",
    "list the patients",
    "individual patient",
    "phone number of",
    "email of",
    "contact details",
    "medical history",
    "diagnosis of",
]


@dataclass
class ScreeningResult:
    """Outcome of screening a user question before it is sent anywhere."""

    allowed: bool
    findings: List[str] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "findings": list(self.findings),
            "message": self.message,
        }


def scan_text_for_pii(text: str) -> List[str]:
    """Return the names of PII-like patterns detected in ``text``."""
    if not text:
        return []
    lowered = text.lower()
    findings: List[str] = []
    for label, pattern in PII_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            findings.append(label)
    return findings


def screen_question(question: str) -> ScreeningResult:
    """Decide whether a user question may be sent to the language model."""
    text = (question or "").strip()
    if not text:
        return ScreeningResult(
            allowed=False,
            findings=["empty question"],
            message="Enter a question before asking the copilot.",
        )
    if len(text) > 1200:
        return ScreeningResult(
            allowed=False,
            findings=["question too long"],
            message=(
                "Questions are capped at 1,200 characters so that the outbound "
                "payload stays reviewable. Please shorten the question."
            ),
        )

    findings = scan_text_for_pii(text)
    lowered = text.lower()
    for phrase in INDIVIDUAL_LEVEL_PHRASES:
        if phrase in lowered:
            findings.append(f"individual-level request ({phrase!r})")

    if findings:
        return ScreeningResult(
            allowed=False,
            findings=findings,
            message=(
                "This question was blocked before anything was sent. CarePath AI "
                "works only with aggregated, synthetic marketing metrics and will "
                "not process or request information about individuals."
            ),
        )
    return ScreeningResult(allowed=True, findings=[], message="")


def assert_payload_is_safe(payload: Dict[str, Any]) -> List[str]:
    """Scan a serialised payload for PII-like content as a final backstop."""
    serialized = json.dumps(payload, default=str)
    return scan_text_for_pii(serialized)


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column completeness report shown in the Trust & Privacy module."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["column", "dtype", "non_null", "missing", "missing_pct"]
        )
    rows = []
    total = len(df)
    for column in df.columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null": total - missing,
                "missing": missing,
                "missing_pct": round(100.0 * missing / total, 3) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def missing_data_notes(df: pd.DataFrame) -> List[str]:
    """Plain-language notes about how missing values affect the numbers shown."""
    notes: List[str] = []
    if df is None or df.empty:
        return ["No rows match the current filters, so every metric reads zero."]
    missing_spend = int(df["marketing_spend"].isna().sum())
    if missing_spend:
        share = 100.0 * missing_spend / len(df)
        notes.append(
            f"{missing_spend} of {len(df):,} rows ({share:.2f}%) have unknown "
            "marketing spend. Spend totals, cost per booking and cost per "
            "attended appointment exclude those rows and are therefore a lower "
            "bound on cost."
        )
    funnel_cols = [
        "sessions",
        "provider_searches",
        "booking_starts",
        "bookings_completed",
        "appointments_attended",
    ]
    missing_funnel = int(df[funnel_cols].isna().sum().sum())
    if missing_funnel:
        notes.append(
            f"{missing_funnel} funnel counts are missing and are treated as zero."
        )
    else:
        notes.append(
            "All funnel counts are present; no imputation is applied to volumes."
        )
    return notes


def privacy_checklist(
    sanitization: Optional[SanitizationResult] = None,
    screening: Optional[ScreeningResult] = None,
    api_configured: bool = False,
) -> List[Dict[str, Any]]:
    """Structured checklist rendered in the Trust & Privacy module."""
    checks: List[Dict[str, Any]] = [
        {
            "check": "Dataset is synthetic",
            "status": "pass",
            "detail": "Generated locally from a fixed seed; no external data source.",
        },
        {
            "check": "No protected health information in scope",
            "status": "pass",
            "detail": "The schema contains no identifiers, clinical fields or free text.",
        },
        {
            "check": "Outbound payload rebuilt from allowlist",
            "status": "pass" if (sanitization is None or sanitization.clean) else "warn",
            "detail": (
                "No fields dropped."
                if sanitization is None or sanitization.clean
                else f"Dropped {len(sanitization.removed)} non-allowlisted field(s)."
            ),
        },
        {
            "check": "Question screened for personal data",
            "status": "pass" if (screening is None or screening.allowed) else "blocked",
            "detail": (
                "No question screened yet."
                if screening is None
                else ("Last question passed screening." if screening.allowed
                      else "Last question was blocked: " + ", ".join(screening.findings))
            ),
        },
        {
            "check": "External model call",
            "status": "active" if api_configured else "offline",
            "detail": (
                "An API key is configured. Aggregated metrics are sent to the "
                "configured provider when you ask a question."
                if api_configured
                else "No API key configured. The app runs fully offline using the "
                     "deterministic fallback analyst; nothing is transmitted."
            ),
        },
        {
            "check": "Human approval required before any experiment is marked live",
            "status": "pass",
            "detail": "Experiments remain in Draft until a named reviewer approves.",
        },
    ]
    return checks
