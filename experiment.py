"""
experiment.py
=============

The AI Experiment Studio.

Turns a recommendation into something a marketing team could actually run: a
hypothesis, a randomised design, a powered sample size, guardrail metrics, an
explicit financial projection, and a human approval gate.

Two things are kept rigorously separate here:

**Measured facts** come from ``analytics.py`` - baseline rates, segment
volumes, observed downstream conversion.

**Projections** are everything that depends on an assumed lift. They are
labelled as projections in every dataclass, every column name and every string
this module produces. A projection is a planning input. It is not a forecast,
and it is certainly not a result.

The simulator is a third category again: it draws synthetic outcomes from an
*assumed* true lift so a reviewer can see how the readout and the statistics
would behave. Running it proves nothing about the real world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from analytics import STEP_DEFINITIONS, safe_div, two_proportion_test

# --------------------------------------------------------------------------
# Approval workflow
# --------------------------------------------------------------------------

STATUS_DRAFT = "Draft"
STATUS_IN_REVIEW = "Pending review"
STATUS_APPROVED = "Approved to run"
STATUS_REJECTED = "Rejected"

ALLOWED_TRANSITIONS: Dict[str, List[str]] = {
    STATUS_DRAFT: [STATUS_IN_REVIEW],
    STATUS_IN_REVIEW: [STATUS_APPROVED, STATUS_REJECTED, STATUS_DRAFT],
    STATUS_APPROVED: [STATUS_DRAFT],
    STATUS_REJECTED: [STATUS_DRAFT],
}

DEFAULT_GUARDRAILS: List[str] = [
    "Appointment attendance rate (a booking nobody keeps is not a win)",
    "No-show rate",
    "Cost per attended appointment",
    "Booking abandonment on the control experience",
]

#: Default share of a measured parity gap an experiment is assumed to capture.
#: Full parity (1.0) is an upper bound that a single change rarely reaches.
DEFAULT_CAPTURE_RATE = 0.30


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------

@dataclass
class BusinessAssumptions:
    """User-adjustable inputs behind every projected figure.

    Every one of these is an assumption, not a measurement. They are surfaced
    in the UI as sliders precisely so a reviewer can see how much the headline
    number depends on them.
    """

    margin_per_attended: float = 280.0
    capture_rate: float = DEFAULT_CAPTURE_RATE
    horizon_days: int = 90
    implementation_cost: float = 0.0
    attendance_rate: float = 0.83
    alpha: float = 0.05
    power: float = 0.80

    def as_dict(self) -> Dict[str, Any]:
        return {
            "margin_per_attended_appointment": round(float(self.margin_per_attended), 2),
            "share_of_gap_assumed_captured": round(float(self.capture_rate), 4),
            "projection_horizon_days": int(self.horizon_days),
            "implementation_cost": round(float(self.implementation_cost), 2),
            "assumed_attendance_rate": round(float(self.attendance_rate), 4),
            "significance_level_alpha": float(self.alpha),
            "statistical_power": float(self.power),
        }


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------

@dataclass
class ExperimentDesign:
    """A testable change, expressed so it can be measured."""

    title: str
    hypothesis: str
    proposed_change: str
    dimension: str
    segment: str
    step_key: str
    step_label: str
    baseline_rate: float
    benchmark_rate: float
    benchmark_segment: str
    eligible_units_in_window: int
    window_days: int
    downstream_to_attended: float
    primary_metric: str
    guardrail_metrics: List[str] = field(default_factory=lambda: list(DEFAULT_GUARDRAILS))
    rationale: str = ""

    @property
    def gap(self) -> float:
        return max(self.benchmark_rate - self.baseline_rate, 0.0)

    @property
    def relative_gap(self) -> float:
        return safe_div(self.gap, self.baseline_rate)

    def target_rate(self, capture_rate: float) -> float:
        """The rate the experiment is designed to reach at a given capture."""
        return self.baseline_rate + self.gap * float(capture_rate)

    def relative_lift(self, capture_rate: float) -> float:
        """Relative improvement implied by capturing ``capture_rate`` of the gap."""
        return safe_div(self.gap * float(capture_rate), self.baseline_rate)

    def eligible_units_per_day(self) -> float:
        return safe_div(self.eligible_units_in_window, max(self.window_days, 1))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "hypothesis": self.hypothesis,
            "proposed_change": self.proposed_change,
            "target_segment": f"{self.dimension} = {self.segment}",
            "step": self.step_label,
            "baseline_rate": round(self.baseline_rate, 6),
            "benchmark_rate": round(self.benchmark_rate, 6),
            "benchmark_segment": self.benchmark_segment,
            "measured_gap": round(self.gap, 6),
            "eligible_units_in_window": int(self.eligible_units_in_window),
            "window_days": int(self.window_days),
            "primary_metric": self.primary_metric,
            "guardrail_metrics": list(self.guardrail_metrics),
            "rationale": self.rationale,
        }


#: Intervention ideas keyed by ``(dimension, step_key)``, falling back to step.
_INTERVENTION_LIBRARY: Dict[str, str] = {
    "device|completion_rate": (
        "Replace the single-page mobile booking form with a three-step flow: "
        "reason for visit, provider and time, then contact details. Add a "
        "progress indicator, native date/time pickers, autofill-friendly field "
        "names, and resume-where-you-left-off on return."
    ),
    "device|start_rate": (
        "Put a persistent 'Book this provider' action in the mobile viewport on "
        "provider and search-results pages, instead of requiring a scroll to "
        "reach it."
    ),
    "channel|start_rate": (
        "Route this channel's traffic to a bookable provider page with "
        "availability visible above the fold, rather than to a general service "
        "or information page."
    ),
    "channel|search_rate": (
        "Match ad creative to a specific service line and land the visitor on "
        "that service's provider list, so the first screen already answers the "
        "intent the ad created."
    ),
    "channel|completion_rate": (
        "Pre-fill what the channel already knows (service line, location) into "
        "the booking flow so this traffic starts the form part-way through."
    ),
    "service_line|start_rate": (
        "Add real appointment availability to provider cards for this service "
        "line so visitors can see a bookable slot before committing to the flow."
    ),
    "service_line|completion_rate": (
        "Remove insurance verification from the booking form for this service "
        "line and move it to a post-booking confirmation step."
    ),
    "*|attendance_rate": (
        "Add a two-stage reminder (SMS at 48 hours, SMS at 2 hours) with "
        "one-tap reschedule, so a conflict produces a moved appointment rather "
        "than a no-show."
    ),
    "*|search_rate": (
        "Surface a prominent provider search entry point on the highest-traffic "
        "landing pages for this segment."
    ),
    "*|start_rate": (
        "Reduce the distance between finding a provider and starting a booking "
        "for this segment to a single action."
    ),
    "*|completion_rate": (
        "Shorten and restructure the booking form for this segment, deferring "
        "every field that is not required to confirm the appointment."
    ),
}


def _intervention_for(dimension: str, step_key: str) -> str:
    return (
        _INTERVENTION_LIBRARY.get(f"{dimension}|{step_key}")
        or _INTERVENTION_LIBRARY.get(f"*|{step_key}")
        or "Reduce friction for this segment at this step."
    )


def recommend_experiments(
    evidence: Dict[str, Any],
    max_experiments: int = 5,
) -> List[ExperimentDesign]:
    """Derive candidate experiments from the ranked opportunities in evidence.

    Nothing here is generated by a language model. Each design is assembled
    deterministically from a measured parity gap plus a human-written
    intervention from :data:`_INTERVENTION_LIBRARY`, so the same evidence always
    yields the same experiment backlog.
    """
    opportunities = [
        o for o in (evidence.get("ranked_opportunities") or []) if isinstance(o, dict)
    ]
    scope = evidence.get("data_scope", {}) or {}
    window_days = int(scope.get("days_in_window") or 90)

    dimension_records = {
        "channel": evidence.get("by_channel") or [],
        "device": evidence.get("by_device") or [],
        "service_line": evidence.get("by_service_line") or [],
    }
    step_keys_by_label = {label: key for key, _, _, label in STEP_DEFINITIONS}

    designs: List[ExperimentDesign] = []
    for opportunity in opportunities[:max_experiments]:
        dimension = str(opportunity.get("dimension", ""))
        segment = str(opportunity.get("segment", ""))
        step_label = str(opportunity.get("step", ""))
        step_key = step_keys_by_label.get(step_label)
        if not step_key or dimension not in dimension_records:
            continue

        definition = next((d for d in STEP_DEFINITIONS if d[0] == step_key), None)
        if definition is None:
            continue
        _, num_field, den_field, _ = definition

        record = next(
            (
                r
                for r in dimension_records[dimension]
                if isinstance(r, dict) and str(r.get(dimension)) == segment
            ),
            None,
        )
        if record is None:
            continue

        eligible = int(record.get(den_field, 0) or 0)
        if eligible <= 0:
            continue

        # Observed downstream conversion from this step through to attendance.
        step_index = [d[0] for d in STEP_DEFINITIONS].index(step_key)
        downstream = 1.0
        for _, d_num, d_den, _ in STEP_DEFINITIONS[step_index + 1:]:
            downstream *= safe_div(record.get(d_num, 0), record.get(d_den, 0))

        baseline = float(opportunity.get("segment_rate", 0.0))
        benchmark = float(opportunity.get("benchmark_rate", 0.0))
        benchmark_segment = str(opportunity.get("benchmark_segment", ""))

        designs.append(
            ExperimentDesign(
                title=f"{segment}: improve {step_label.lower()}",
                hypothesis=(
                    f"If we reduce friction for {dimension.replace('_', ' ')} "
                    f"'{segment}' at the '{step_label.lower()}' step, that step's "
                    f"conversion rate will rise from {baseline * 100:.1f}% toward "
                    f"the {benchmark * 100:.1f}% observed for '{benchmark_segment}', "
                    f"without reducing appointment attendance."
                ),
                proposed_change=_intervention_for(dimension, step_key),
                dimension=dimension,
                segment=segment,
                step_key=step_key,
                step_label=step_label,
                baseline_rate=baseline,
                benchmark_rate=benchmark,
                benchmark_segment=benchmark_segment,
                eligible_units_in_window=eligible,
                window_days=window_days,
                downstream_to_attended=downstream,
                primary_metric=f"{step_label} rate for {segment}",
                rationale=(
                    f"'{segment}' converts at {baseline * 100:.1f}% at this step "
                    f"against {benchmark * 100:.1f}% for '{benchmark_segment}', on "
                    f"{eligible:,} eligible units over {window_days} days. The gap "
                    f"is measured; the mechanism behind it is not."
                ),
            )
        )
    return designs


# --------------------------------------------------------------------------
# Power analysis
# --------------------------------------------------------------------------

def sample_size_per_arm(
    baseline_rate: float,
    relative_lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Units required **per arm** to detect ``relative_lift`` on ``baseline_rate``.

    Standard two-proportion, two-sided sample-size formula. Returns ``0`` when
    the inputs make the question meaningless (zero lift, degenerate baseline).
    """
    p1 = float(baseline_rate)
    if p1 <= 0.0 or p1 >= 1.0 or relative_lift <= 0.0:
        return 0
    p2 = min(p1 * (1.0 + float(relative_lift)), 0.999999)
    delta = p2 - p1
    if delta <= 0:
        return 0

    pooled = (p1 + p2) / 2.0
    z_alpha = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    z_beta = float(stats.norm.ppf(float(power)))

    numerator = (
        z_alpha * math.sqrt(2.0 * pooled * (1.0 - pooled))
        + z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return int(math.ceil(numerator / (delta ** 2)))


def runtime_estimate_days(
    sample_per_arm: int,
    eligible_units_per_day: float,
    arms: int = 2,
    traffic_allocation: float = 1.0,
) -> float:
    """How long the design needs to run, given daily eligible traffic."""
    per_day = float(eligible_units_per_day) * float(traffic_allocation)
    if per_day <= 0 or sample_per_arm <= 0:
        return 0.0
    return (sample_per_arm * arms) / per_day


# --------------------------------------------------------------------------
# Financial projection
# --------------------------------------------------------------------------

@dataclass
class ImpactProjection:
    """A projected financial case. Every field is a projection, not a result."""

    baseline_rate: float
    target_rate: float
    absolute_lift: float
    relative_lift: float
    eligible_units_per_day: float
    horizon_days: int
    projected_incremental_conversions: float
    projected_incremental_attended: float
    margin_per_attended: float
    projected_incremental_margin: float
    implementation_cost: float
    projected_net_margin: float
    projected_annualised_margin: float
    sample_size_per_arm: int
    estimated_runtime_days: float
    assumptions: Dict[str, Any] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "baseline_rate": round(self.baseline_rate, 6),
            "projected_target_rate": round(self.target_rate, 6),
            "projected_absolute_lift": round(self.absolute_lift, 6),
            "projected_relative_lift": round(self.relative_lift, 6),
            "eligible_units_per_day": round(self.eligible_units_per_day, 2),
            "horizon_days": int(self.horizon_days),
            "projected_incremental_conversions": round(
                self.projected_incremental_conversions, 1
            ),
            "projected_incremental_attended_appointments": round(
                self.projected_incremental_attended, 1
            ),
            "margin_per_attended_appointment": round(self.margin_per_attended, 2),
            "projected_incremental_margin": round(self.projected_incremental_margin, 2),
            "implementation_cost": round(self.implementation_cost, 2),
            "projected_net_margin": round(self.projected_net_margin, 2),
            "projected_annualised_margin": round(self.projected_annualised_margin, 2),
            "sample_size_per_arm": int(self.sample_size_per_arm),
            "estimated_runtime_days": round(self.estimated_runtime_days, 1),
            "assumptions": self.assumptions,
            "caveats": list(self.caveats),
        }


def project_impact(
    design: ExperimentDesign,
    assumptions: BusinessAssumptions,
    traffic_allocation: float = 1.0,
) -> ImpactProjection:
    """Build the projected financial case for a design under stated assumptions.

    The chain is deliberately simple and fully inspectable:

    ``eligible units/day x horizon x absolute lift`` -> incremental conversions
    -> ``x observed downstream conversion`` -> incremental attended appointments
    -> ``x margin per attended appointment`` -> incremental contribution margin.
    """
    capture = float(assumptions.capture_rate)
    target = design.target_rate(capture)
    absolute_lift = max(target - design.baseline_rate, 0.0)
    relative_lift = design.relative_lift(capture)

    per_day = design.eligible_units_per_day()
    horizon = int(assumptions.horizon_days)

    incremental_conversions = per_day * horizon * absolute_lift
    downstream = float(design.downstream_to_attended)
    if design.step_key == "attendance_rate":
        downstream = 1.0
    incremental_attended = incremental_conversions * downstream

    margin_each = float(assumptions.margin_per_attended)
    incremental_margin = incremental_attended * margin_each
    net_margin = incremental_margin - float(assumptions.implementation_cost)
    annualised = safe_div(incremental_margin, horizon) * 365.0

    n_per_arm = sample_size_per_arm(
        design.baseline_rate,
        relative_lift,
        alpha=assumptions.alpha,
        power=assumptions.power,
    )
    runtime = runtime_estimate_days(
        n_per_arm, per_day, arms=2, traffic_allocation=traffic_allocation
    )

    caveats = [
        "PROJECTION, NOT A FORECAST. Every figure below assumes the experiment "
        f"captures {capture * 100:.0f}% of the measured gap to "
        f"'{design.benchmark_segment}'. That share is an assumption and has not "
        "been tested.",
        "The lift is applied to the primary step only. Downstream conversion is "
        "held at this segment's observed rates, which assumes the added "
        "conversions behave like the existing ones - they may be lower intent.",
        f"Contribution margin of {margin_each:,.0f} per attended appointment is a "
        "planning assumption for a fictional health system, not a measured value.",
        "The annualised figure simply scales the horizon result and ignores "
        "seasonality, competitive response and novelty decay.",
        "No implementation risk, engineering cost or opportunity cost of "
        "alternative work is modelled unless entered above.",
    ]
    if 0 < runtime < 14:
        caveats.append(
            f"The powered sample size is reached in about {runtime:,.1f} days, but "
            "a run that short will not cover a full weekly cycle. Plan at least "
            "two weeks so weekday and weekend behaviour are both represented."
        )
    if runtime > horizon:
        caveats.append(
            f"The powered sample size needs about {runtime:,.0f} days of traffic, "
            f"which is longer than the {horizon}-day projection horizon. Either "
            "accept a larger detectable effect or plan a longer run."
        )

    return ImpactProjection(
        baseline_rate=design.baseline_rate,
        target_rate=target,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        eligible_units_per_day=per_day,
        horizon_days=horizon,
        projected_incremental_conversions=incremental_conversions,
        projected_incremental_attended=incremental_attended,
        margin_per_attended=margin_each,
        projected_incremental_margin=incremental_margin,
        implementation_cost=float(assumptions.implementation_cost),
        projected_net_margin=net_margin,
        projected_annualised_margin=annualised,
        sample_size_per_arm=n_per_arm,
        estimated_runtime_days=runtime,
        assumptions=assumptions.as_dict(),
        caveats=caveats,
    )


# --------------------------------------------------------------------------
# Synthetic experiment simulator
# --------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Outcome of a synthetic A/B run. Demonstrates readout mechanics only."""

    n_per_arm: int
    control_conversions: int
    treatment_conversions: int
    control_rate: float
    treatment_rate: float
    observed_absolute_lift: float
    observed_relative_lift: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool
    true_relative_lift: float
    seed: int
    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_per_arm": self.n_per_arm,
            "control_conversions": self.control_conversions,
            "treatment_conversions": self.treatment_conversions,
            "control_rate": round(self.control_rate, 6),
            "treatment_rate": round(self.treatment_rate, 6),
            "observed_absolute_lift": round(self.observed_absolute_lift, 6),
            "observed_relative_lift": round(self.observed_relative_lift, 6),
            "p_value": float(f"{self.p_value:.6g}"),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "significant": bool(self.significant),
            "assumed_true_relative_lift": round(self.true_relative_lift, 6),
            "seed": self.seed,
        }


def simulate_experiment(
    design: ExperimentDesign,
    true_relative_lift: float,
    n_per_arm: int,
    alpha: float = 0.05,
    seed: int = 7,
) -> SimulationResult:
    """Draw one synthetic A/B outcome from an assumed true lift.

    This is a teaching device, not evidence. It shows what the readout would
    look like if the assumed lift were real, and - usefully - how often a real
    lift still fails to reach significance at a given sample size.
    """
    n_per_arm = int(max(n_per_arm, 1))
    control_rate = float(min(max(design.baseline_rate, 1e-6), 0.999999))
    treatment_rate = float(min(control_rate * (1.0 + float(true_relative_lift)), 0.999999))

    rng = np.random.default_rng(int(seed))
    control_conversions = int(rng.binomial(n_per_arm, control_rate))
    treatment_conversions = int(rng.binomial(n_per_arm, treatment_rate))

    test = two_proportion_test(
        treatment_conversions, n_per_arm, control_conversions, n_per_arm, alpha=alpha
    )

    caveats = [
        "SYNTHETIC SIMULATION. Outcomes are drawn from an assumed true lift, so "
        "this result is evidence about the *design*, not about the product.",
        "A single draw is one possible world. Change the seed to see how much "
        "the readout moves at this sample size.",
        "Statistical significance is not business significance; check the "
        "confidence interval against the smallest lift worth shipping.",
        "Guardrail metrics are not simulated here and must be monitored in any "
        "real run.",
    ]
    if not test.significant and true_relative_lift > 0:
        caveats.append(
            "This run did not reach significance even though a real lift was "
            "assumed - a direct illustration of why stopping early is unsafe."
        )

    return SimulationResult(
        n_per_arm=n_per_arm,
        control_conversions=control_conversions,
        treatment_conversions=treatment_conversions,
        control_rate=safe_div(control_conversions, n_per_arm),
        treatment_rate=safe_div(treatment_conversions, n_per_arm),
        observed_absolute_lift=test.difference,
        observed_relative_lift=test.relative_difference,
        p_value=test.p_value,
        ci_low=test.ci_low,
        ci_high=test.ci_high,
        significant=test.significant,
        true_relative_lift=float(true_relative_lift),
        seed=int(seed),
        caveats=caveats,
    )


# --------------------------------------------------------------------------
# Experiment record and human approval
# --------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """A design plus its projection, its approval state and its audit trail."""

    design: ExperimentDesign
    assumptions: BusinessAssumptions
    projection: ImpactProjection
    status: str = STATUS_DRAFT
    reviewer: str = ""
    review_notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    history: List[Dict[str, str]] = field(default_factory=list)
    source: str = "AI recommendation"

    def transition(
        self,
        new_status: str,
        reviewer: str = "",
        notes: str = "",
    ) -> bool:
        """Move the record to ``new_status`` if that transition is permitted."""
        if new_status not in ALLOWED_TRANSITIONS.get(self.status, []):
            return False
        if new_status in (STATUS_APPROVED, STATUS_REJECTED) and not reviewer.strip():
            return False
        previous = self.status
        self.status = new_status
        if reviewer.strip():
            self.reviewer = reviewer.strip()
        if notes.strip():
            self.review_notes = notes.strip()
        self.history.append(
            {
                "from": previous,
                "to": new_status,
                "reviewer": self.reviewer or "unassigned",
                "notes": notes.strip(),
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
        return True

    @property
    def is_approved(self) -> bool:
        return self.status == STATUS_APPROVED

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reviewer": self.reviewer or None,
            "review_notes": self.review_notes or None,
            "created_at": self.created_at,
            "source": self.source,
            "design": self.design.as_dict(),
            "projection": self.projection.as_dict(),
            "history": list(self.history),
        }


def build_record(
    design: ExperimentDesign,
    assumptions: BusinessAssumptions,
    traffic_allocation: float = 1.0,
    source: str = "AI recommendation",
) -> ExperimentRecord:
    """Create a Draft experiment record with its projection attached."""
    return ExperimentRecord(
        design=design,
        assumptions=assumptions,
        projection=project_impact(design, assumptions, traffic_allocation),
        source=source,
    )


def portfolio_summary(records: Sequence[ExperimentRecord]) -> Dict[str, Any]:
    """Roll up a set of experiment records for the backlog view."""
    approved = [r for r in records if r.is_approved]
    return {
        "total": len(records),
        "approved": len(approved),
        "pending_review": sum(1 for r in records if r.status == STATUS_IN_REVIEW),
        "rejected": sum(1 for r in records if r.status == STATUS_REJECTED),
        "projected_margin_all": round(
            sum(r.projection.projected_incremental_margin for r in records), 2
        ),
        "projected_margin_approved": round(
            sum(r.projection.projected_incremental_margin for r in approved), 2
        ),
        "note": (
            "Projected values assume each experiment delivers its assumed share "
            "of the measured gap. Overlapping experiments are not additive."
        ),
    }
