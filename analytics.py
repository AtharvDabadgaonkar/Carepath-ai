"""
analytics.py
============

Deterministic analytics for CarePath AI.

Every number the product shows - in the dashboard, in an AI answer, or in an
experiment projection - is produced by a function in this module. The AI layer
is never asked to compute anything; it only receives the output of these
functions as structured evidence and is asked to interpret it.

That separation is deliberate. It means an AI answer can be audited: the user
can open the "model input" panel, see the exact aggregated numbers that were
sent, and re-derive them from the dashboard.

Conventions
-----------
* All rates are returned as fractions in ``[0, 1]`` unless the function name
  ends in ``_pct``.
* Division by zero returns ``0.0`` rather than ``NaN`` or an exception.
* ``marketing_spend`` may contain nulls by design (see ``data_generator``).
  Sums skip nulls, and every function that touches spend also reports how much
  of it was unknown so the user is never shown a silently understated total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from data_generator import (
    CONTRIBUTION_MARGIN,
    FUNNEL_COLUMNS,
    FUNNEL_LABELS,
)

# --------------------------------------------------------------------------
# Funnel step definitions
# --------------------------------------------------------------------------

#: ``(key, numerator_stage, denominator_stage, human label)`` for each step.
STEP_DEFINITIONS: List[Tuple[str, str, str, str]] = [
    ("search_rate", "provider_searches", "sessions",
     "Session to provider search"),
    ("start_rate", "booking_starts", "provider_searches",
     "Provider search to booking start"),
    ("completion_rate", "bookings_completed", "booking_starts",
     "Booking start to booking completed"),
    ("attendance_rate", "appointments_attended", "bookings_completed",
     "Booking completed to appointment attended"),
]

STEP_LABELS: Dict[str, str] = {k: lbl for k, _, _, lbl in STEP_DEFINITIONS}

DIMENSIONS: List[str] = ["channel", "device", "service_line"]

DIMENSION_LABELS: Dict[str, str] = {
    "channel": "Acquisition channel",
    "device": "Device",
    "service_line": "Service line",
}

#: Segments below this many sessions are excluded from benchmark selection so
#: a tiny, noisy segment cannot become the "best in class" target.
MIN_SEGMENT_SESSIONS = 500


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` instead of raising or producing NaN."""
    try:
        if denominator is None or float(denominator) == 0.0:
            return default
        value = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[object, object]] = None,
    channels: Optional[Sequence[str]] = None,
    devices: Optional[Sequence[str]] = None,
    service_lines: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return the subset of ``df`` matching the supplied filters.

    Empty or ``None`` filter lists mean "no restriction on this dimension",
    which matches how the Streamlit multiselects behave when cleared.
    """
    out = df
    if date_range and date_range[0] is not None and date_range[1] is not None:
        start = pd.Timestamp(date_range[0]).normalize()
        end = pd.Timestamp(date_range[1]).normalize()
        if start > end:
            start, end = end, start
        out = out[(out["date"] >= start) & (out["date"] <= end)]
    if channels:
        out = out[out["channel"].isin(list(channels))]
    if devices:
        out = out[out["device"].isin(list(devices))]
    if service_lines:
        out = out[out["service_line"].isin(list(service_lines))]
    return out.copy()


# --------------------------------------------------------------------------
# Headline KPIs
# --------------------------------------------------------------------------

def kpi_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Headline metrics for the KPI cards and for the AI evidence pack."""
    if df.empty:
        empty = {c: 0 for c in FUNNEL_COLUMNS}
        empty.update(
            {
                "marketing_spend": 0.0,
                "contribution_margin": 0.0,
                "session_to_booking_rate": 0.0,
                "session_to_attended_rate": 0.0,
                "cost_per_booking": 0.0,
                "cost_per_attended": 0.0,
                "margin_per_session": 0.0,
                "return_on_spend": 0.0,
                "no_show_rate": 0.0,
                "spend_rows_missing": 0,
                "spend_missing_share": 0.0,
                "days": 0,
            }
        )
        for key, _, _, _ in STEP_DEFINITIONS:
            empty[key] = 0.0
        return empty

    totals = {c: int(df[c].sum()) for c in FUNNEL_COLUMNS}
    spend = float(df["marketing_spend"].sum(skipna=True))
    margin = float(df["estimated_contribution_margin"].sum(skipna=True))
    missing_rows = int(df["marketing_spend"].isna().sum())

    summary: Dict[str, float] = dict(totals)
    summary["marketing_spend"] = round(spend, 2)
    summary["contribution_margin"] = round(margin, 2)
    summary["session_to_booking_rate"] = safe_div(
        totals["bookings_completed"], totals["sessions"]
    )
    summary["session_to_attended_rate"] = safe_div(
        totals["appointments_attended"], totals["sessions"]
    )
    summary["cost_per_booking"] = safe_div(spend, totals["bookings_completed"])
    summary["cost_per_attended"] = safe_div(spend, totals["appointments_attended"])
    summary["margin_per_session"] = safe_div(margin, totals["sessions"])
    summary["return_on_spend"] = safe_div(margin, spend)
    summary["no_show_rate"] = 1.0 - safe_div(
        totals["appointments_attended"], totals["bookings_completed"]
    )
    summary["spend_rows_missing"] = missing_rows
    summary["spend_missing_share"] = safe_div(missing_rows, len(df))
    summary["days"] = int(df["date"].nunique())

    for key, num, den, _ in STEP_DEFINITIONS:
        summary[key] = safe_div(totals[num], totals[den])

    return summary


# --------------------------------------------------------------------------
# Funnel
# --------------------------------------------------------------------------

def funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Blended funnel: stage volumes, step conversion and absolute drop-off."""
    totals = {c: int(df[c].sum()) if not df.empty else 0 for c in FUNNEL_COLUMNS}
    rows = []
    previous: Optional[int] = None
    top = totals[FUNNEL_COLUMNS[0]]
    for stage in FUNNEL_COLUMNS:
        count = totals[stage]
        step_rate = 1.0 if previous is None else safe_div(count, previous)
        drop = 0 if previous is None else previous - count
        rows.append(
            {
                "stage": stage,
                "stage_label": FUNNEL_LABELS[stage],
                "count": count,
                "step_conversion": step_rate,
                "drop_off": drop,
                "drop_off_rate": 0.0 if previous is None else safe_div(drop, previous),
                "share_of_sessions": safe_div(count, top),
            }
        )
        previous = count
    return pd.DataFrame(rows)


def biggest_leak(df: pd.DataFrame) -> Dict[str, object]:
    """Identify the funnel step that loses the most absolute volume."""
    funnel = funnel_counts(df)
    candidates = funnel.iloc[1:]
    if candidates.empty or candidates["drop_off"].max() <= 0:
        return {
            "step_label": "n/a",
            "from_stage": "",
            "to_stage": "",
            "lost": 0,
            "drop_off_rate": 0.0,
            "step_conversion": 0.0,
        }
    row = candidates.loc[candidates["drop_off"].idxmax()]
    idx = FUNNEL_COLUMNS.index(str(row["stage"]))
    from_stage = FUNNEL_COLUMNS[idx - 1]
    step_key = STEP_DEFINITIONS[idx - 1][0]
    return {
        "step_key": step_key,
        "step_label": STEP_LABELS[step_key],
        "from_stage": FUNNEL_LABELS[from_stage],
        "to_stage": FUNNEL_LABELS[str(row["stage"])],
        "lost": int(row["drop_off"]),
        "drop_off_rate": float(row["drop_off_rate"]),
        "step_conversion": float(row["step_conversion"]),
    }


def step_rates_by(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Per-segment funnel volumes and step conversion rates."""
    if dimension not in DIMENSIONS:
        raise KeyError(f"Unsupported dimension: {dimension}")
    if df.empty:
        return pd.DataFrame(columns=[dimension, *FUNNEL_COLUMNS])

    grouped = df.groupby(dimension, observed=True)[FUNNEL_COLUMNS].sum()
    spend = df.groupby(dimension, observed=True)["marketing_spend"].sum(min_count=0)
    margin = df.groupby(dimension, observed=True)[
        "estimated_contribution_margin"
    ].sum(min_count=0)

    out = grouped.copy()
    out["marketing_spend"] = spend
    out["contribution_margin"] = margin
    for key, num, den, _ in STEP_DEFINITIONS:
        out[key] = out.apply(lambda r, n=num, d=den: safe_div(r[n], r[d]), axis=1)
    out["session_to_booking_rate"] = out.apply(
        lambda r: safe_div(r["bookings_completed"], r["sessions"]), axis=1
    )
    out["session_to_attended_rate"] = out.apply(
        lambda r: safe_div(r["appointments_attended"], r["sessions"]), axis=1
    )
    out["cost_per_attended"] = out.apply(
        lambda r: safe_div(r["marketing_spend"], r["appointments_attended"]), axis=1
    )
    out["return_on_spend"] = out.apply(
        lambda r: safe_div(r["contribution_margin"], r["marketing_spend"]), axis=1
    )
    return out.reset_index().sort_values("sessions", ascending=False)


def channel_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Spend efficiency per acquisition channel."""
    table = step_rates_by(df, "channel")
    if table.empty:
        return table
    cols = [
        "channel",
        "sessions",
        "bookings_completed",
        "appointments_attended",
        "marketing_spend",
        "contribution_margin",
        "session_to_booking_rate",
        "cost_per_attended",
        "return_on_spend",
    ]
    return table[cols].sort_values("marketing_spend", ascending=False)


def daily_trend(df: pd.DataFrame, dimension: Optional[str] = None) -> pd.DataFrame:
    """Daily (optionally per-segment) funnel volumes and conversion rate."""
    if df.empty:
        return pd.DataFrame(columns=["date", *FUNNEL_COLUMNS])
    keys = ["date"] + ([dimension] if dimension else [])
    out = df.groupby(keys, observed=True)[FUNNEL_COLUMNS].sum().reset_index()
    out["session_to_booking_rate"] = out.apply(
        lambda r: safe_div(r["bookings_completed"], r["sessions"]), axis=1
    )
    out["completion_rate"] = out.apply(
        lambda r: safe_div(r["bookings_completed"], r["booking_starts"]), axis=1
    )
    return out


# --------------------------------------------------------------------------
# Statistical testing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProportionTest:
    """Result of a two-proportion z-test."""

    rate_a: float
    rate_b: float
    n_a: int
    n_b: int
    difference: float
    relative_difference: float
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    alpha: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "rate_a": round(self.rate_a, 6),
            "rate_b": round(self.rate_b, 6),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "difference": round(self.difference, 6),
            "relative_difference": round(self.relative_difference, 6),
            "z_statistic": round(self.z_statistic, 4),
            "p_value": float(f"{self.p_value:.6g}"),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "significant": bool(self.significant),
            "alpha": self.alpha,
        }


def two_proportion_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    alpha: float = 0.05,
) -> ProportionTest:
    """Two-sided two-proportion z-test with a Wald confidence interval.

    ``a`` is treated as the comparison group and ``b`` as the reference, so a
    negative ``difference`` means group ``a`` converts worse than group ``b``.
    """
    trials_a = int(max(trials_a, 0))
    trials_b = int(max(trials_b, 0))
    successes_a = int(min(max(successes_a, 0), trials_a))
    successes_b = int(min(max(successes_b, 0), trials_b))

    p_a = safe_div(successes_a, trials_a)
    p_b = safe_div(successes_b, trials_b)
    diff = p_a - p_b

    if trials_a == 0 or trials_b == 0:
        return ProportionTest(
            p_a, p_b, trials_a, trials_b, diff, 0.0, 0.0, 1.0, 0.0, 0.0, False, alpha
        )

    pooled = safe_div(successes_a + successes_b, trials_a + trials_b)
    se_pooled = math.sqrt(
        max(pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b), 1e-12)
    )
    z = safe_div(diff, se_pooled)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_unpooled = math.sqrt(
        max(p_a * (1 - p_a) / trials_a + p_b * (1 - p_b) / trials_b, 1e-12)
    )
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    return ProportionTest(
        rate_a=p_a,
        rate_b=p_b,
        n_a=trials_a,
        n_b=trials_b,
        difference=diff,
        relative_difference=safe_div(diff, p_b),
        z_statistic=z,
        p_value=p_value,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


def segment_step_test(
    df: pd.DataFrame,
    dimension: str,
    step_key: str,
    segment_a: str,
    segment_b: str,
    alpha: float = 0.05,
) -> ProportionTest:
    """Compare one funnel step between two segments of a dimension."""
    definition = next((d for d in STEP_DEFINITIONS if d[0] == step_key), None)
    if definition is None:
        raise KeyError(f"Unknown step: {step_key}")
    _, num, den, _ = definition
    grouped = df.groupby(dimension, observed=True)[[num, den]].sum()
    row_a = grouped.loc[segment_a] if segment_a in grouped.index else None
    row_b = grouped.loc[segment_b] if segment_b in grouped.index else None
    if row_a is None or row_b is None:
        return two_proportion_test(0, 0, 0, 0, alpha)
    return two_proportion_test(
        int(row_a[num]), int(row_a[den]), int(row_b[num]), int(row_b[den]), alpha
    )


# --------------------------------------------------------------------------
# Opportunity sizing
# --------------------------------------------------------------------------

def _weighted_margin(df: pd.DataFrame, mask: Optional[pd.Series] = None) -> float:
    """Contribution margin per attended appointment for a slice of the data."""
    subset = df if mask is None else df[mask]
    attended = float(subset["appointments_attended"].sum())
    if attended <= 0:
        return float(np.mean(list(CONTRIBUTION_MARGIN.values())))
    return safe_div(
        float(subset["estimated_contribution_margin"].sum()),
        attended,
        default=float(np.mean(list(CONTRIBUTION_MARGIN.values()))),
    )


def opportunity_table(
    df: pd.DataFrame,
    dimension: str,
    step_key: str,
    capture_rate: float = 1.0,
    margin_overrides: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Size the gap between each segment and the best-performing peer.

    For the chosen funnel step, the highest-converting segment with at least
    ``MIN_SEGMENT_SESSIONS`` sessions becomes the benchmark. For every other
    segment the function estimates how many additional conversions would occur
    at benchmark parity, flows them through that segment's *observed*
    downstream rates to attended appointments, and values them at that
    segment's observed contribution margin per attended appointment.

    ``capture_rate`` scales the gap: ``1.0`` is full parity (an upper bound
    that no single change is likely to reach), ``0.3`` is a more conservative
    "capture 30% of the gap" planning scenario.

    The result is a **projection under stated assumptions**, not a forecast and
    not a measured outcome.
    """
    definition = next((d for d in STEP_DEFINITIONS if d[0] == step_key), None)
    if definition is None:
        raise KeyError(f"Unknown step: {step_key}")
    _, num, den, step_label = definition

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(dimension, observed=True)[FUNNEL_COLUMNS].sum()
    if grouped.empty:
        return pd.DataFrame()

    grouped["step_rate"] = grouped.apply(lambda r: safe_div(r[num], r[den]), axis=1)

    eligible = grouped[grouped["sessions"] >= MIN_SEGMENT_SESSIONS]
    pool = eligible if not eligible.empty else grouped
    benchmark_segment = str(pool["step_rate"].idxmax())
    benchmark_rate = float(pool["step_rate"].max())

    step_index = [d[0] for d in STEP_DEFINITIONS].index(step_key)
    downstream = STEP_DEFINITIONS[step_index + 1:]

    rows = []
    for segment, row in grouped.iterrows():
        segment = str(segment)
        rate = float(row["step_rate"])
        gap = max(benchmark_rate - rate, 0.0)
        extra_conversions = gap * float(row[den]) * float(capture_rate)

        # Flow the extra conversions through this segment's own downstream rates.
        flowed = extra_conversions
        for _, d_num, d_den, _ in downstream:
            flowed *= safe_div(row[d_num], row[d_den])

        if margin_overrides:
            mask = df[dimension] == segment
            attended = float(df.loc[mask, "appointments_attended"].sum())
            margin_val = (
                safe_div(
                    float(
                        (
                            df.loc[mask, "appointments_attended"]
                            * df.loc[mask, "service_line"].map(
                                lambda s: margin_overrides.get(
                                    s, CONTRIBUTION_MARGIN.get(s, 0.0)
                                )
                            )
                        ).sum()
                    ),
                    attended,
                )
                if attended > 0
                else 0.0
            )
        else:
            margin_val = _weighted_margin(df, df[dimension] == segment)

        rows.append(
            {
                dimension: segment,
                "denominator": int(row[den]),
                "conversions": int(row[num]),
                "step_rate": rate,
                "benchmark_segment": benchmark_segment,
                "benchmark_rate": benchmark_rate,
                "gap": gap,
                "relative_gap": safe_div(gap, benchmark_rate),
                "recoverable_conversions": extra_conversions,
                "recoverable_attended": flowed,
                "margin_per_attended": margin_val,
                "opportunity_value": flowed * margin_val,
                "step_label": step_label,
                "capture_rate": float(capture_rate),
            }
        )

    out = pd.DataFrame(rows).sort_values("opportunity_value", ascending=False)
    return out.reset_index(drop=True)


def ranked_opportunities(
    df: pd.DataFrame,
    capture_rate: float = 1.0,
    top_n: int = 6,
) -> pd.DataFrame:
    """Rank every (dimension, segment, step) parity gap by projected value."""
    frames = []
    for dimension in DIMENSIONS:
        for step_key, _, _, _ in STEP_DEFINITIONS:
            table = opportunity_table(df, dimension, step_key, capture_rate)
            if table.empty:
                continue
            table = table.rename(columns={dimension: "segment"})
            table.insert(0, "dimension", dimension)
            frames.append(table)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["gap"] > 0]
    combined = combined.sort_values("opportunity_value", ascending=False)
    return combined.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Period comparison
# --------------------------------------------------------------------------

def period_comparison(df: pd.DataFrame) -> Dict[str, object]:
    """Compare the most recent half of the window with the prior half."""
    if df.empty or df["date"].nunique() < 4:
        return {"available": False}

    dates = np.sort(df["date"].unique())
    split = dates[len(dates) // 2]
    recent = df[df["date"] >= split]
    prior = df[df["date"] < split]

    recent_kpi = kpi_summary(recent)
    prior_kpi = kpi_summary(prior)

    deltas = {}
    for metric in (
        "sessions",
        "bookings_completed",
        "appointments_attended",
        "marketing_spend",
        "session_to_booking_rate",
        "cost_per_attended",
    ):
        before = float(prior_kpi.get(metric, 0.0))
        after = float(recent_kpi.get(metric, 0.0))
        deltas[metric] = {
            "prior": round(before, 4),
            "recent": round(after, 4),
            "absolute_change": round(after - before, 4),
            "relative_change": round(safe_div(after - before, before), 4),
        }

    return {
        "available": True,
        "split_date": str(pd.Timestamp(split).date()),
        "prior_days": int(prior["date"].nunique()),
        "recent_days": int(recent["date"].nunique()),
        "metrics": deltas,
    }


# --------------------------------------------------------------------------
# Evidence pack: the only thing the AI layer ever sees
# --------------------------------------------------------------------------

@dataclass
class EvidencePack:
    """Aggregated, de-identified business metrics prepared for the AI layer."""

    payload: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return self.payload


def _round_rates(mapping: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def build_evidence_pack(
    df: pd.DataFrame,
    filters_applied: Optional[Dict[str, object]] = None,
    capture_rate: float = 1.0,
) -> Dict[str, object]:
    """Build the aggregated evidence dictionary used by the AI copilot.

    This is the *entire* universe of facts the language model receives. It
    contains only aggregated, business-level metrics computed by this module -
    no row-level records, no identifiers, and no free text from the dataset.
    """
    summary = _round_rates(kpi_summary(df))
    funnel = funnel_counts(df)
    leak = biggest_leak(df)

    by_dimension: Dict[str, object] = {}
    for dimension in DIMENSIONS:
        table = step_rates_by(df, dimension)
        if table.empty:
            by_dimension[dimension] = []
            continue
        keep = [
            dimension,
            "sessions",
            "provider_searches",
            "booking_starts",
            "bookings_completed",
            "appointments_attended",
            "marketing_spend",
            "contribution_margin",
            "search_rate",
            "start_rate",
            "completion_rate",
            "attendance_rate",
            "session_to_booking_rate",
            "cost_per_attended",
            "return_on_spend",
        ]
        records = table[keep].to_dict(orient="records")
        by_dimension[dimension] = [_round_rates(r) for r in records]

    device_test = segment_step_test(
        df, "device", "completion_rate", "Mobile", "Desktop"
    )

    opportunities = ranked_opportunities(df, capture_rate=capture_rate, top_n=6)
    opportunity_records = []
    if not opportunities.empty:
        for record in opportunities.to_dict(orient="records"):
            opportunity_records.append(
                _round_rates(
                    {
                        "dimension": record["dimension"],
                        "segment": record["segment"],
                        "step": record["step_label"],
                        "segment_rate": record["step_rate"],
                        "benchmark_segment": record["benchmark_segment"],
                        "benchmark_rate": record["benchmark_rate"],
                        "gap": record["gap"],
                        "recoverable_attended_appointments": round(
                            float(record["recoverable_attended"]), 1
                        ),
                        "projected_margin_opportunity": round(
                            float(record["opportunity_value"]), 2
                        ),
                        "capture_rate_assumption": record["capture_rate"],
                    }
                )
            )

    payload: Dict[str, object] = {
        "data_scope": {
            "is_synthetic": True,
            "days_in_window": int(summary.get("days", 0)),
            "start_date": str(pd.Timestamp(df["date"].min()).date())
            if not df.empty
            else None,
            "end_date": str(pd.Timestamp(df["date"].max()).date())
            if not df.empty
            else None,
            "rows_aggregated": int(len(df)),
            "filters_applied": filters_applied or {},
        },
        "kpi_summary": summary,
        "funnel": [
            _round_rates(r)
            for r in funnel[
                ["stage_label", "count", "step_conversion", "drop_off", "drop_off_rate"]
            ].to_dict(orient="records")
        ],
        "largest_drop_off": _round_rates(dict(leak)),
        "by_channel": by_dimension.get("channel", []),
        "by_device": by_dimension.get("device", []),
        "by_service_line": by_dimension.get("service_line", []),
        "mobile_vs_desktop_completion_test": device_test.as_dict(),
        "ranked_opportunities": opportunity_records,
        "period_comparison": period_comparison(df),
        "data_quality": {
            "spend_rows_missing": int(summary.get("spend_rows_missing", 0)),
            "spend_missing_share": float(summary.get("spend_missing_share", 0.0)),
            "note": (
                "Marketing spend totals exclude rows with unknown spend and are "
                "therefore a lower bound."
            ),
        },
    }
    return payload


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json
    from data_generator import load_dataset

    frame = load_dataset()
    print(json.dumps(build_evidence_pack(frame)["kpi_summary"], indent=2))
