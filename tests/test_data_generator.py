"""Tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_generator as dg


def test_shape_and_grain(dataset):
    expected = (
        dg.DEFAULT_DAYS * len(dg.CHANNELS) * len(dg.DEVICES) * len(dg.SERVICE_LINES)
    )
    assert len(dataset) == expected
    assert dataset["date"].nunique() == dg.DEFAULT_DAYS
    assert set(dataset["channel"]) == set(dg.CHANNELS)
    assert set(dataset["device"]) == set(dg.DEVICES)
    assert set(dataset["service_line"]) == set(dg.SERVICE_LINES)
    assert not dataset.duplicated(
        subset=["date", "channel", "device", "service_line"]
    ).any()


def test_determinism_same_seed():
    a = dg.load_dataset()
    b = dg.load_dataset()
    pd.testing.assert_frame_equal(a, b)
    assert dg.dataset_signature(a) == dg.dataset_signature(b)


def test_different_seed_changes_data():
    a = dg.generate_marketing_data(seed=1)
    b = dg.generate_marketing_data(seed=2)
    assert dg.dataset_signature(a) != dg.dataset_signature(b)


def test_funnel_is_internally_consistent(dataset):
    """No stage may ever exceed the stage above it, row by row."""
    columns = dg.FUNNEL_COLUMNS
    for upper, lower in zip(columns, columns[1:]):
        assert (dataset[lower] <= dataset[upper]).all(), f"{lower} exceeded {upper}"


def test_no_negative_values(dataset):
    numeric = dataset.select_dtypes(include=[np.number])
    assert (numeric.fillna(0) >= 0).all().all()


def test_margin_matches_service_line(dataset):
    for line, margin in dg.CONTRIBUTION_MARGIN.items():
        subset = dataset[dataset["service_line"] == line]
        assert (subset["margin_per_attended"] == margin).all()
        expected = subset["appointments_attended"] * margin
        assert np.allclose(subset["estimated_contribution_margin"], expected)


def test_missing_spend_only_on_paid_channels(dataset):
    missing = dataset[dataset["marketing_spend"].isna()]
    assert len(missing) > 0, "the generator should inject some missing spend"
    assert set(missing["channel"]).issubset(set(dg.PAID_CHANNELS))
    assert len(missing) / len(dataset) < 0.05


def test_unpaid_channels_have_no_spend(dataset):
    unpaid = dataset[~dataset["channel"].isin(dg.PAID_CHANNELS)]
    assert (unpaid["marketing_spend"].fillna(0) == 0).all()


def test_planted_mobile_completion_gap(dataset):
    grouped = dataset.groupby("device")[["booking_starts", "bookings_completed"]].sum()
    rates = grouped["bookings_completed"] / grouped["booking_starts"]
    assert rates["Desktop"] - rates["Mobile"] > 0.15


def test_inject_missing_can_be_disabled():
    frame = dg.generate_marketing_data(inject_missing=False)
    assert frame["marketing_spend"].isna().sum() == 0


def test_rejects_bad_day_count():
    with pytest.raises(ValueError):
        dg.generate_marketing_data(days=0)


def test_describe_dataset(dataset):
    meta = dg.describe_dataset(dataset)
    assert meta.synthetic is True
    assert meta.rows == len(dataset)
    assert len(meta.signature) == 12


def test_margin_assumptions_override():
    overrides = dg.margin_assumptions({"Cardiology": 999.0, "Nonexistent": 1.0})
    assert overrides["Cardiology"] == 999.0
    assert "Nonexistent" not in overrides
    assert overrides["Primary Care"] == dg.CONTRIBUTION_MARGIN["Primary Care"]
