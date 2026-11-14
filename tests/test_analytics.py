"""Tests for the deterministic analytics layer."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import analytics
from data_generator import FUNNEL_COLUMNS, load_dataset


def test_safe_div_handles_zero_and_garbage():
    assert analytics.safe_div(10, 2) == 5.0
    assert analytics.safe_div(1, 0) == 0.0
    assert analytics.safe_div(1, 0, default=-1.0) == -1.0
    assert analytics.safe_div(None, 5) == 0.0
    assert analytics.safe_div("x", 5) == 0.0


def test_apply_filters_narrows_each_dimension(dataset):
    filtered = analytics.apply_filters(
        dataset, channels=["Email"], devices=["Mobile"],
        service_lines=["Cardiology"],
    )
    assert set(filtered["channel"]) == {"Email"}
    assert set(filtered["device"]) == {"Mobile"}
    assert set(filtered["service_line"]) == {"Cardiology"}
    assert len(filtered) == dataset["date"].nunique()


def test_apply_filters_empty_lists_mean_no_restriction(dataset):
    assert len(analytics.apply_filters(dataset, channels=[], devices=[])) == len(dataset)


def test_apply_filters_date_range_and_reversed_range(dataset):
    start = pd.Timestamp(dataset["date"].min())
    end = start + pd.Timedelta(days=6)
    forward = analytics.apply_filters(dataset, date_range=(start, end))
    backward = analytics.apply_filters(dataset, date_range=(end, start))
    assert forward["date"].nunique() == 7
    assert len(forward) == len(backward)


def test_kpi_summary_totals_match_raw_sums(dataset):
    summary = analytics.kpi_summary(dataset)
    for column in FUNNEL_COLUMNS:
        assert summary[column] == int(dataset[column].sum())
    assert summary["session_to_booking_rate"] == pytest.approx(
        dataset["bookings_completed"].sum() / dataset["sessions"].sum()
    )


def test_kpi_summary_on_empty_frame_is_all_zero(dataset):
    summary = analytics.kpi_summary(dataset.iloc[0:0])
    assert summary["sessions"] == 0
    assert summary["session_to_booking_rate"] == 0.0
    assert summary["cost_per_attended"] == 0.0


def test_kpi_summary_reports_missing_spend(dataset):
    summary = analytics.kpi_summary(dataset)
    assert summary["spend_rows_missing"] == int(dataset["marketing_spend"].isna().sum())
    assert 0.0 <= summary["spend_missing_share"] < 1.0


def test_funnel_counts_are_monotonic(dataset):
    funnel = analytics.funnel_counts(dataset)
    counts = funnel["count"].tolist()
    assert counts == sorted(counts, reverse=True)
    assert funnel.iloc[0]["step_conversion"] == 1.0


def test_funnel_drop_off_adds_up(dataset):
    funnel = analytics.funnel_counts(dataset)
    for i in range(1, len(funnel)):
        previous = funnel.iloc[i - 1]["count"]
        assert funnel.iloc[i]["count"] + funnel.iloc[i]["drop_off"] == previous


def test_biggest_leak_identifies_a_real_step(dataset):
    leak = analytics.biggest_leak(dataset)
    assert leak["lost"] > 0
    assert leak["step_key"] in {k for k, _, _, _ in analytics.STEP_DEFINITIONS}


def test_biggest_leak_on_empty_frame(dataset):
    leak = analytics.biggest_leak(dataset.iloc[0:0])
    assert leak["lost"] == 0


def test_step_rates_by_covers_every_segment(dataset):
    for dimension in analytics.DIMENSIONS:
        table = analytics.step_rates_by(dataset, dimension)
        assert len(table) == dataset[dimension].nunique()
        for key, _, _, _ in analytics.STEP_DEFINITIONS:
            assert table[key].between(0, 1).all()


def test_step_rates_by_rejects_unknown_dimension(dataset):
    with pytest.raises(KeyError):
        analytics.step_rates_by(dataset, "not_a_dimension")


def test_two_proportion_test_detects_a_real_difference():
    result = analytics.two_proportion_test(400, 1000, 500, 1000)
    assert result.significant
    assert result.difference == pytest.approx(-0.1)
    assert result.ci_low < result.difference < result.ci_high


def test_two_proportion_test_on_identical_groups():
    result = analytics.two_proportion_test(500, 1000, 500, 1000)
    assert not result.significant
    assert result.p_value == pytest.approx(1.0)


def test_two_proportion_test_handles_empty_groups():
    result = analytics.two_proportion_test(0, 0, 0, 0)
    assert not result.significant
    assert result.p_value == 1.0


def test_mobile_desktop_gap_is_significant(dataset):
    result = analytics.segment_step_test(
        dataset, "device", "completion_rate", "Mobile", "Desktop"
    )
    assert result.significant
    assert result.difference < 0


def test_segment_step_test_unknown_segment_is_safe(dataset):
    result = analytics.segment_step_test(
        dataset, "device", "completion_rate", "Smartwatch", "Desktop"
    )
    assert result.n_a == 0


def test_opportunity_table_benchmark_has_no_gap(dataset):
    table = analytics.opportunity_table(dataset, "device", "completion_rate")
    benchmark = table.iloc[0]["benchmark_segment"]
    row = table[table["device"] == benchmark].iloc[0]
    assert row["gap"] == pytest.approx(0.0)
    assert row["opportunity_value"] == pytest.approx(0.0)


def test_opportunity_scales_linearly_with_capture_rate(dataset):
    full = analytics.opportunity_table(dataset, "device", "completion_rate", 1.0)
    half = analytics.opportunity_table(dataset, "device", "completion_rate", 0.5)
    full_value = full["opportunity_value"].max()
    half_value = half["opportunity_value"].max()
    assert half_value == pytest.approx(full_value / 2, rel=1e-6)


def test_opportunity_table_rejects_unknown_step(dataset):
    with pytest.raises(KeyError):
        analytics.opportunity_table(dataset, "device", "not_a_step")


def test_ranked_opportunities_sorted_and_positive(dataset):
    ranked = analytics.ranked_opportunities(dataset, top_n=6)
    values = ranked["opportunity_value"].tolist()
    assert values == sorted(values, reverse=True)
    assert all(v > 0 for v in values)
    assert (ranked["gap"] > 0).all()


def test_daily_trend_one_row_per_day(dataset):
    trend = analytics.daily_trend(dataset)
    assert len(trend) == dataset["date"].nunique()
    by_device = analytics.daily_trend(dataset, "device")
    assert len(by_device) == dataset["date"].nunique() * dataset["device"].nunique()


def test_period_comparison_splits_the_window(dataset):
    comparison = analytics.period_comparison(dataset)
    assert comparison["available"]
    assert comparison["prior_days"] + comparison["recent_days"] == (
        dataset["date"].nunique()
    )
    assert "cost_per_attended" in comparison["metrics"]


def test_period_comparison_needs_enough_days(dataset):
    tiny = dataset[dataset["date"] <= dataset["date"].min()]
    assert analytics.period_comparison(tiny)["available"] is False


def test_evidence_pack_structure(evidence):
    for section in (
        "data_scope", "kpi_summary", "funnel", "largest_drop_off",
        "by_channel", "by_device", "by_service_line",
        "mobile_vs_desktop_completion_test", "ranked_opportunities",
        "period_comparison", "data_quality",
    ):
        assert section in evidence
    assert evidence["data_scope"]["is_synthetic"] is True


def test_evidence_pack_is_json_serialisable(evidence):
    import json

    text = json.dumps(evidence)
    assert len(text) > 1000


def test_evidence_pack_contains_no_nan(evidence):
    def walk(node):
        if isinstance(node, float):
            assert not math.isnan(node) and not math.isinf(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(evidence)


def test_evidence_pack_handles_empty_selection(dataset):
    pack = analytics.build_evidence_pack(dataset.iloc[0:0])
    assert pack["kpi_summary"]["sessions"] == 0
    assert pack["ranked_opportunities"] == []
