"""Tests for experiment design, power, projection, simulation and approvals."""

from __future__ import annotations

import pytest

import experiment
from experiment import BusinessAssumptions, ExperimentDesign


@pytest.fixture()
def designs(evidence):
    return experiment.recommend_experiments(evidence)


@pytest.fixture()
def design(designs):
    return designs[0]


@pytest.fixture()
def assumptions():
    return BusinessAssumptions(
        margin_per_attended=280.0, capture_rate=0.30, horizon_days=90,
        implementation_cost=0.0,
    )


# ------------------------------------------------------------------- designs

def test_recommendations_are_produced(designs):
    assert len(designs) >= 3
    for item in designs:
        assert isinstance(item, ExperimentDesign)
        assert item.eligible_units_in_window > 0
        assert 0 < item.baseline_rate < 1
        assert item.benchmark_rate >= item.baseline_rate


def test_recommendations_are_deterministic(evidence):
    first = experiment.recommend_experiments(evidence)
    second = experiment.recommend_experiments(evidence)
    assert [d.as_dict() for d in first] == [d.as_dict() for d in second]


def test_top_recommendation_is_the_mobile_booking_gap(design):
    assert design.segment == "Mobile"
    assert design.step_key == "completion_rate"


def test_every_design_has_guardrails(designs):
    for item in designs:
        assert len(item.guardrail_metrics) >= 3
        assert any("attend" in g.lower() for g in item.guardrail_metrics)


def test_design_hypothesis_mentions_the_benchmark(design):
    assert design.benchmark_segment in design.hypothesis


def test_no_recommendations_from_empty_evidence():
    assert experiment.recommend_experiments({}) == []


def test_target_rate_scales_with_capture(design):
    assert design.target_rate(0.0) == pytest.approx(design.baseline_rate)
    assert design.target_rate(1.0) == pytest.approx(design.benchmark_rate)
    midpoint = design.target_rate(0.5)
    assert design.baseline_rate < midpoint < design.benchmark_rate


# --------------------------------------------------------------------- power

def test_sample_size_shrinks_as_the_effect_grows():
    small = experiment.sample_size_per_arm(0.44, 0.05)
    large = experiment.sample_size_per_arm(0.44, 0.20)
    assert small > large > 0


def test_sample_size_grows_with_power():
    assert (
        experiment.sample_size_per_arm(0.44, 0.10, power=0.95)
        > experiment.sample_size_per_arm(0.44, 0.10, power=0.80)
    )


def test_sample_size_grows_as_alpha_tightens():
    assert (
        experiment.sample_size_per_arm(0.44, 0.10, alpha=0.01)
        > experiment.sample_size_per_arm(0.44, 0.10, alpha=0.10)
    )


@pytest.mark.parametrize("baseline,lift", [(0.0, 0.1), (1.0, 0.1), (0.44, 0.0)])
def test_sample_size_degenerate_inputs_return_zero(baseline, lift):
    assert experiment.sample_size_per_arm(baseline, lift) == 0


def test_runtime_estimate_scales_with_traffic():
    assert experiment.runtime_estimate_days(1000, 500) == pytest.approx(4.0)
    assert experiment.runtime_estimate_days(1000, 0) == 0.0
    assert experiment.runtime_estimate_days(0, 500) == 0.0


# ---------------------------------------------------------------- projection

def test_projection_chain_is_arithmetically_consistent(design, assumptions):
    p = experiment.project_impact(design, assumptions)
    expected_conversions = (
        p.eligible_units_per_day * p.horizon_days * p.absolute_lift
    )
    assert p.projected_incremental_conversions == pytest.approx(expected_conversions)
    assert p.projected_incremental_attended == pytest.approx(
        expected_conversions * design.downstream_to_attended
    )
    assert p.projected_incremental_margin == pytest.approx(
        p.projected_incremental_attended * assumptions.margin_per_attended
    )


def test_projection_scales_linearly_with_capture_rate(design):
    low = experiment.project_impact(
        design, BusinessAssumptions(capture_rate=0.20)
    ).projected_incremental_margin
    high = experiment.project_impact(
        design, BusinessAssumptions(capture_rate=0.40)
    ).projected_incremental_margin
    assert high == pytest.approx(low * 2, rel=1e-6)


def test_zero_capture_means_zero_opportunity(design):
    p = experiment.project_impact(design, BusinessAssumptions(capture_rate=0.0))
    assert p.projected_incremental_margin == pytest.approx(0.0)
    assert p.sample_size_per_arm == 0


def test_implementation_cost_reduces_net_margin(design):
    with_cost = experiment.project_impact(
        design, BusinessAssumptions(implementation_cost=50_000.0)
    )
    assert with_cost.projected_net_margin == pytest.approx(
        with_cost.projected_incremental_margin - 50_000.0
    )


def test_every_projection_is_labelled_as_one(design, assumptions):
    p = experiment.project_impact(design, assumptions)
    assert p.caveats
    assert "PROJECTION, NOT A FORECAST" in p.caveats[0]
    assert any("assumption" in c.lower() for c in p.caveats)
    assert all(
        key.startswith("projected_") or "projected" in key or key in {
            "baseline_rate", "eligible_units_per_day", "horizon_days",
            "margin_per_attended_appointment", "implementation_cost",
            "sample_size_per_arm", "estimated_runtime_days", "assumptions",
            "caveats",
        }
        for key in p.as_dict()
    )


def test_short_runtime_triggers_a_weekly_cycle_caveat(design):
    p = experiment.project_impact(design, BusinessAssumptions(capture_rate=1.0))
    if 0 < p.estimated_runtime_days < 14:
        assert any("weekly cycle" in c for c in p.caveats)


def test_assumptions_are_recorded_in_the_projection(design, assumptions):
    recorded = experiment.project_impact(design, assumptions).assumptions
    assert recorded["margin_per_attended_appointment"] == 280.0
    assert recorded["share_of_gap_assumed_captured"] == 0.30


# ---------------------------------------------------------------- simulation

def test_simulation_is_reproducible(design):
    a = experiment.simulate_experiment(design, 0.15, 5000, seed=11)
    b = experiment.simulate_experiment(design, 0.15, 5000, seed=11)
    assert a.as_dict() == b.as_dict()


def test_simulation_seed_changes_the_draw(design):
    a = experiment.simulate_experiment(design, 0.15, 5000, seed=1)
    b = experiment.simulate_experiment(design, 0.15, 5000, seed=2)
    assert a.treatment_conversions != b.treatment_conversions


def test_large_true_lift_is_detected(design):
    result = experiment.simulate_experiment(design, 0.40, 20000, seed=3)
    assert result.significant
    assert result.observed_relative_lift > 0


def test_zero_lift_usually_is_not_significant(design):
    """With no true effect, a well-powered test should mostly not fire."""
    firings = sum(
        experiment.simulate_experiment(design, 0.0, 8000, seed=s).significant
        for s in range(20)
    )
    assert firings <= 3


def test_simulation_always_carries_caveats(design):
    result = experiment.simulate_experiment(design, 0.1, 3000, seed=5)
    assert "SYNTHETIC SIMULATION" in result.caveats[0]
    assert len(result.caveats) >= 4


def test_simulation_handles_tiny_sample(design):
    result = experiment.simulate_experiment(design, 0.1, 1, seed=5)
    assert result.n_per_arm == 1


# ------------------------------------------------------------------ approvals

def test_record_starts_as_draft(design, assumptions):
    record = experiment.build_record(design, assumptions)
    assert record.status == experiment.STATUS_DRAFT
    assert not record.is_approved
    assert record.history == []


def test_happy_path_approval(design, assumptions):
    record = experiment.build_record(design, assumptions)
    assert record.transition(experiment.STATUS_IN_REVIEW)
    assert record.transition(experiment.STATUS_APPROVED, "A. Reviewer", "looks sound")
    assert record.is_approved
    assert record.reviewer == "A. Reviewer"
    assert len(record.history) == 2
    assert record.history[-1]["to"] == experiment.STATUS_APPROVED


def test_approval_requires_a_named_reviewer(design, assumptions):
    record = experiment.build_record(design, assumptions)
    record.transition(experiment.STATUS_IN_REVIEW)
    assert not record.transition(experiment.STATUS_APPROVED, "   ")
    assert record.status == experiment.STATUS_IN_REVIEW


def test_cannot_skip_review(design, assumptions):
    record = experiment.build_record(design, assumptions)
    assert not record.transition(experiment.STATUS_APPROVED, "A. Reviewer")
    assert record.status == experiment.STATUS_DRAFT


def test_rejected_can_return_to_draft(design, assumptions):
    record = experiment.build_record(design, assumptions)
    record.transition(experiment.STATUS_IN_REVIEW)
    record.transition(experiment.STATUS_REJECTED, "A. Reviewer", "not now")
    assert record.transition(experiment.STATUS_DRAFT)
    assert record.status == experiment.STATUS_DRAFT


def test_record_serialises_for_export(design, assumptions):
    import json

    record = experiment.build_record(design, assumptions)
    text = json.dumps(record.as_dict(), default=str)
    assert "projection" in text and "design" in text


def test_portfolio_summary_counts_only_approved(design, assumptions):
    approved = experiment.build_record(design, assumptions)
    approved.transition(experiment.STATUS_IN_REVIEW)
    approved.transition(experiment.STATUS_APPROVED, "A. Reviewer")
    draft = experiment.build_record(design, assumptions)

    summary = experiment.portfolio_summary([approved, draft])
    assert summary["total"] == 2
    assert summary["approved"] == 1
    assert summary["projected_margin_approved"] == pytest.approx(
        approved.projection.projected_incremental_margin, rel=1e-6
    )
    assert "not additive" in summary["note"]


def test_portfolio_summary_on_empty_list():
    assert experiment.portfolio_summary([])["total"] == 0
