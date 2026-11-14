"""
data_generator.py
=================

Deterministic synthetic data generation for CarePath AI.

This module fabricates 90 days of *fictional* healthcare website marketing
activity. Nothing here is derived from real patients, real visitors, or any
real organisation's analytics. The dataset is generated from a fixed seed so
that every reviewer who runs this project sees exactly the same numbers.

Grain of the dataset
--------------------
One row per ``(date, channel, device, service_line)`` combination:

    90 days x 6 channels x 2 devices x 5 service lines = 5,400 rows

Funnel modelled
---------------
    sessions
        -> provider_searches      (visitor searched for a provider)
        -> booking_starts         (visitor opened the booking flow)
        -> bookings_completed     (visitor confirmed an appointment)
        -> appointments_attended  (patient actually showed up)

Each stage is drawn with ``numpy.random.Generator.binomial`` from the previous
stage, which guarantees the funnel is internally consistent (no stage can ever
exceed the stage above it).

Deliberately planted signals
----------------------------
The synthetic world contains a small number of patterns that the analytics and
AI layers are expected to rediscover from the data alone:

1. **Mobile booking-completion gap** - mobile visitors complete the booking
   form at roughly 43% versus roughly 71% on desktop. This is the single
   largest source of lost appointments.
2. **Paid Social quality problem** - Paid Social buys a lot of cheap, mostly
   mobile sessions with weak downstream intent, so its cost per attended
   appointment is the worst of any channel.
3. **A mid-period campaign ramp** - Paid Social spend steps up around day 46,
   which dilutes blended conversion rate in the second half of the window.
4. **Weekday seasonality** - weekends carry far less traffic.
5. **A small amount of missing spend data** - about 1% of paid rows have a
   null ``marketing_spend`` so that the missing-data handling in the rest of
   the application has something real to handle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_DAYS = 90
DEFAULT_END_DATE = "2025-09-30"

#: Ordered funnel stages. ``analytics.py`` relies on this ordering.
FUNNEL_COLUMNS: List[str] = [
    "sessions",
    "provider_searches",
    "booking_starts",
    "bookings_completed",
    "appointments_attended",
]

FUNNEL_LABELS: Dict[str, str] = {
    "sessions": "Website sessions",
    "provider_searches": "Provider searches",
    "booking_starts": "Booking starts",
    "bookings_completed": "Bookings completed",
    "appointments_attended": "Appointments attended",
}

CHANNELS: List[str] = [
    "Organic Search",
    "Paid Search",
    "Paid Social",
    "Direct",
    "Email",
    "Referral",
]

PAID_CHANNELS: List[str] = ["Paid Search", "Paid Social", "Email"]

DEVICES: List[str] = ["Mobile", "Desktop"]

SERVICE_LINES: List[str] = [
    "Primary Care",
    "Cardiology",
    "Orthopedics",
    "OB-GYN",
    "Dermatology",
]

#: Estimated contribution margin per *attended* appointment, in USD.
#: These are illustrative planning assumptions for a fictional health system,
#: not published benchmarks. They are user-adjustable in the Experiment Studio.
CONTRIBUTION_MARGIN: Dict[str, float] = {
    "Primary Care": 185.0,
    "Cardiology": 430.0,
    "Orthopedics": 495.0,
    "OB-GYN": 265.0,
    "Dermatology": 225.0,
}

# Share of total sessions by acquisition channel.
_CHANNEL_SESSION_SHARE: Dict[str, float] = {
    "Organic Search": 0.30,
    "Paid Search": 0.18,
    "Paid Social": 0.16,
    "Direct": 0.17,
    "Email": 0.10,
    "Referral": 0.09,
}

# Share of a channel's sessions that arrive on mobile.
_CHANNEL_MOBILE_SHARE: Dict[str, float] = {
    "Organic Search": 0.62,
    "Paid Search": 0.66,
    "Paid Social": 0.82,
    "Direct": 0.55,
    "Email": 0.58,
    "Referral": 0.48,
}

# Multipliers applied to each funnel step, by channel. Values above 1.0 mean
# the channel converts better than the blended baseline at that step.
_CHANNEL_STEP_QUALITY: Dict[str, Dict[str, float]] = {
    "Organic Search": {"search": 1.12, "start": 1.08, "complete": 1.02, "attend": 1.01},
    "Paid Search":    {"search": 1.04, "start": 1.02, "complete": 1.00, "attend": 1.00},
    "Paid Social":    {"search": 0.69, "start": 0.62, "complete": 0.94, "attend": 0.93},
    "Direct":         {"search": 1.18, "start": 1.14, "complete": 1.04, "attend": 1.03},
    "Email":          {"search": 1.22, "start": 1.26, "complete": 1.05, "attend": 1.05},
    "Referral":       {"search": 1.09, "start": 1.11, "complete": 1.03, "attend": 1.04},
}

# Blended baseline step rates before channel / device adjustment.
_BASE_SEARCH_RATE = 0.47
_BASE_START_RATE = 0.40

# The planted headline signal: booking-form completion on mobile is far worse
# than on desktop. Attendance is also modestly worse on mobile.
_COMPLETE_RATE_BY_DEVICE: Dict[str, float] = {"Desktop": 0.71, "Mobile": 0.43}
_ATTEND_RATE_BY_DEVICE: Dict[str, float] = {"Desktop": 0.84, "Mobile": 0.79}

# Mobile visitors browse slightly more but intend slightly less.
_DEVICE_SEARCH_MULT: Dict[str, float] = {"Desktop": 0.97, "Mobile": 1.02}
_DEVICE_START_MULT: Dict[str, float] = {"Desktop": 1.06, "Mobile": 0.96}

# Service lines differ in how much research a visitor does before booking.
_SERVICE_LINE_SHARE: Dict[str, float] = {
    "Primary Care": 0.34,
    "Cardiology": 0.14,
    "Orthopedics": 0.17,
    "OB-GYN": 0.18,
    "Dermatology": 0.17,
}

_SERVICE_LINE_STEP_QUALITY: Dict[str, Dict[str, float]] = {
    "Primary Care": {"search": 1.06, "start": 1.05, "complete": 1.03, "attend": 1.00},
    "Cardiology":   {"search": 0.94, "start": 0.90, "complete": 0.95, "attend": 1.04},
    "Orthopedics":  {"search": 0.96, "start": 0.93, "complete": 0.97, "attend": 1.02},
    "OB-GYN":       {"search": 1.02, "start": 1.02, "complete": 1.01, "attend": 1.01},
    "Dermatology":  {"search": 1.03, "start": 1.07, "complete": 1.02, "attend": 0.97},
}

# Day-of-week traffic multipliers (Monday == 0).
_WEEKDAY_FACTOR: List[float] = [1.13, 1.10, 1.06, 1.02, 0.94, 0.62, 0.73]

# Cost per session by channel, in USD. Organic / Direct / Referral are unpaid.
_COST_PER_SESSION: Dict[str, float] = {
    "Organic Search": 0.0,
    "Paid Search": 3.75,
    "Paid Social": 1.55,
    "Direct": 0.0,
    "Email": 0.09,
    "Referral": 0.0,
}

_BASE_DAILY_SESSIONS = 5_200
_TREND_PER_DAY = 0.0018          # gentle organic growth over the window
_SOCIAL_RAMP_DAY = 46            # Paid Social campaign steps up here
_SOCIAL_RAMP_MULTIPLIER = 1.65
_MISSING_SPEND_RATE = 0.012      # fraction of paid rows with null spend


@dataclass(frozen=True)
class DatasetMeta:
    """Provenance describing how a dataset instance was produced."""

    seed: int
    days: int
    end_date: str
    rows: int
    signature: str
    synthetic: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "days": self.days,
            "end_date": self.end_date,
            "rows": self.rows,
            "signature": self.signature,
            "synthetic": self.synthetic,
        }


def _clip_rate(values: np.ndarray) -> np.ndarray:
    """Keep probabilities inside a sane range so binomial draws stay valid."""
    return np.clip(values, 0.01, 0.97)


def _build_skeleton(days: int, end_date: str) -> pd.DataFrame:
    """Create the full (date x channel x device x service_line) grid."""
    end = pd.Timestamp(end_date).normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, CHANNELS, DEVICES, SERVICE_LINES],
        names=["date", "channel", "device", "service_line"],
    )
    frame = index.to_frame(index=False)
    frame["day_index"] = (frame["date"] - frame["date"].min()).dt.days
    frame["weekday"] = frame["date"].dt.weekday
    return frame


def generate_marketing_data(
    seed: int = DEFAULT_SEED,
    days: int = DEFAULT_DAYS,
    end_date: str = DEFAULT_END_DATE,
    inject_missing: bool = True,
) -> pd.DataFrame:
    """Generate the synthetic marketing funnel dataset.

    Parameters
    ----------
    seed:
        Seed for ``numpy.random.default_rng``. The same seed always produces a
        byte-identical dataset.
    days:
        Number of consecutive days to generate.
    end_date:
        Last date in the window (inclusive), as an ISO date string.
    inject_missing:
        When ``True`` a small share of paid rows get a null ``marketing_spend``
        so downstream missing-data handling is exercised.

    Returns
    -------
    pandas.DataFrame
        One row per date/channel/device/service line with funnel counts,
        marketing spend and estimated contribution margin.
    """
    if days < 1:
        raise ValueError("days must be >= 1")

    rng = np.random.default_rng(seed)
    df = _build_skeleton(days, end_date)
    n = len(df)

    # ---------------- session volume ----------------
    channel_share = df["channel"].map(_CHANNEL_SESSION_SHARE).to_numpy()
    device_share = np.where(
        df["device"].to_numpy() == "Mobile",
        df["channel"].map(_CHANNEL_MOBILE_SHARE).to_numpy(),
        1.0 - df["channel"].map(_CHANNEL_MOBILE_SHARE).to_numpy(),
    )
    line_share = df["service_line"].map(_SERVICE_LINE_SHARE).to_numpy()
    weekday_factor = np.array([_WEEKDAY_FACTOR[w] for w in df["weekday"].to_numpy()])
    trend = 1.0 + _TREND_PER_DAY * df["day_index"].to_numpy()

    # Paid Social steps its budget up partway through the window.
    social_ramp = np.where(
        (df["channel"].to_numpy() == "Paid Social")
        & (df["day_index"].to_numpy() >= _SOCIAL_RAMP_DAY),
        _SOCIAL_RAMP_MULTIPLIER,
        1.0,
    )

    lam = (
        _BASE_DAILY_SESSIONS
        * channel_share
        * device_share
        * line_share
        * weekday_factor
        * trend
        * social_ramp
    )
    lam = lam * rng.normal(loc=1.0, scale=0.06, size=n).clip(0.70, 1.30)
    sessions = rng.poisson(np.maximum(lam, 0.0))

    # ---------------- step conversion rates ----------------
    def step_mult(step: str) -> np.ndarray:
        ch = np.array([_CHANNEL_STEP_QUALITY[c][step] for c in df["channel"]])
        sl = np.array([_SERVICE_LINE_STEP_QUALITY[s][step] for s in df["service_line"]])
        return ch * sl

    noise = lambda scale: rng.normal(1.0, scale, size=n).clip(0.80, 1.20)  # noqa: E731

    device_arr = df["device"].to_numpy()

    search_rate = _clip_rate(
        _BASE_SEARCH_RATE
        * step_mult("search")
        * np.array([_DEVICE_SEARCH_MULT[d] for d in device_arr])
        * noise(0.05)
    )
    start_rate = _clip_rate(
        _BASE_START_RATE
        * step_mult("start")
        * np.array([_DEVICE_START_MULT[d] for d in device_arr])
        * noise(0.05)
    )
    complete_rate = _clip_rate(
        np.array([_COMPLETE_RATE_BY_DEVICE[d] for d in device_arr])
        * step_mult("complete")
        * noise(0.04)
    )
    attend_rate = _clip_rate(
        np.array([_ATTEND_RATE_BY_DEVICE[d] for d in device_arr])
        * step_mult("attend")
        * noise(0.03)
    )

    # ---------------- funnel draws ----------------
    provider_searches = rng.binomial(sessions, search_rate)
    booking_starts = rng.binomial(provider_searches, start_rate)
    bookings_completed = rng.binomial(booking_starts, complete_rate)
    appointments_attended = rng.binomial(bookings_completed, attend_rate)

    df["sessions"] = sessions
    df["provider_searches"] = provider_searches
    df["booking_starts"] = booking_starts
    df["bookings_completed"] = bookings_completed
    df["appointments_attended"] = appointments_attended

    # ---------------- money ----------------
    cps = df["channel"].map(_COST_PER_SESSION).to_numpy()
    spend = sessions * cps * rng.normal(1.0, 0.07, size=n).clip(0.75, 1.25)
    # The Paid Social ramp costs more per session as the audience widens.
    spend = spend * np.where(social_ramp > 1.0, 1.18, 1.0)
    df["marketing_spend"] = np.round(spend, 2)

    margin_per_appt = df["service_line"].map(CONTRIBUTION_MARGIN).to_numpy()
    df["margin_per_attended"] = margin_per_appt
    df["estimated_contribution_margin"] = np.round(
        appointments_attended * margin_per_appt, 2
    )

    # ---------------- deliberate missing values ----------------
    if inject_missing:
        paid_mask = df["channel"].isin(PAID_CHANNELS).to_numpy()
        paid_positions = np.flatnonzero(paid_mask)
        n_missing = int(round(len(paid_positions) * _MISSING_SPEND_RATE))
        if n_missing > 0:
            chosen = rng.choice(paid_positions, size=n_missing, replace=False)
            df.loc[df.index[chosen], "marketing_spend"] = np.nan

    df = df.drop(columns=["weekday"])
    column_order = [
        "date",
        "day_index",
        "channel",
        "device",
        "service_line",
        *FUNNEL_COLUMNS,
        "marketing_spend",
        "margin_per_attended",
        "estimated_contribution_margin",
    ]
    return df[column_order].sort_values(
        ["date", "channel", "device", "service_line"]
    ).reset_index(drop=True)


def dataset_signature(df: pd.DataFrame) -> str:
    """Short stable hash of the funnel counts, used to prove determinism."""
    payload = df[FUNNEL_COLUMNS].to_numpy(dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def describe_dataset(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    end_date: str = DEFAULT_END_DATE,
) -> DatasetMeta:
    """Build a :class:`DatasetMeta` provenance record for a generated frame."""
    return DatasetMeta(
        seed=seed,
        days=int(df["date"].nunique()),
        end_date=str(pd.Timestamp(df["date"].max()).date()) or end_date,
        rows=int(len(df)),
        signature=dataset_signature(df),
    )


def load_dataset(
    seed: int = DEFAULT_SEED,
    days: int = DEFAULT_DAYS,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Convenience wrapper used by the app and the tests."""
    return generate_marketing_data(seed=seed, days=days, end_date=end_date)


def dimension_values(df: pd.DataFrame, dimension: str) -> List[str]:
    """Sorted unique values for a categorical dimension."""
    if dimension not in df.columns:
        raise KeyError(f"Unknown dimension: {dimension}")
    return sorted(str(v) for v in df[dimension].dropna().unique())


def margin_assumptions(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return contribution-margin assumptions, optionally overridden by a user."""
    base = dict(CONTRIBUTION_MARGIN)
    if overrides:
        for key, value in overrides.items():
            if key in base:
                base[key] = float(value)
    return base


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    frame = load_dataset()
    meta = describe_dataset(frame)
    print(frame.head())
    print(meta.as_dict())
