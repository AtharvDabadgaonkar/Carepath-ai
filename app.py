"""
app.py
======

CarePath AI - Healthcare Conversion Intelligence.

Streamlit entry point. This file is presentation only: it reads from
``data_generator``, computes through ``analytics``, asks ``ai_analyst``,
designs through ``experiment``, gates on ``privacy`` and grades with
``evaluation``. No business logic lives here.

Run locally:

    streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import evaluation
import experiment
import privacy
from ai_analyst import (
    SUGGESTED_QUESTIONS,
    AIAnalyst,
    AnalystResponse,
    fmt_money,
    fmt_pct,
    mask_key,
    resolve_api_key,
    resolve_model,
)
from data_generator import (
    CHANNELS,
    CONTRIBUTION_MARGIN,
    DEFAULT_DAYS,
    DEFAULT_SEED,
    DEVICES,
    SERVICE_LINES,
    describe_dataset,
    load_dataset,
)

# --------------------------------------------------------------------------
# Brand palette
# --------------------------------------------------------------------------

NAVY = "#0A2540"
NAVY_SOFT = "#13395E"
TEAL = "#0E7C86"
TEAL_LIGHT = "#2FA8A1"
TEAL_PALE = "#E6F3F4"
SLATE = "#5A6B7B"
BORDER = "#E3E9EF"
AMBER = "#B7791F"
CRIMSON = "#B3303C"
SURFACE = "#F7FAFC"

SEQUENCE = [NAVY, TEAL, TEAL_LIGHT, NAVY_SOFT, "#7FB3B8", "#9AA8B6"]

PAGES = [
    "Growth Dashboard",
    "AI Conversion Copilot",
    "AI Experiment Studio",
    "AI Trust & Privacy",
    "About This Project",
]

st.set_page_config(
    page_title="CarePath AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
  .stApp {{ background: #FFFFFF; }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

  h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 650; letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.9rem; }}
  h2 {{ font-size: 1.35rem; margin-top: 0.4rem; }}
  h3 {{ font-size: 1.08rem; }}

  .cp-masthead {{
    background: linear-gradient(104deg, {NAVY} 0%, {NAVY_SOFT} 58%, {TEAL} 140%);
    color: #FFFFFF; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
  }}
  .cp-masthead h1 {{ color: #FFFFFF; margin: 0 0 0.25rem 0; font-size: 1.75rem; }}
  .cp-masthead p {{ margin: 0; color: #CFE3E6; font-size: 0.95rem; }}
  .cp-masthead .cp-chips {{ margin-top: 0.85rem; }}
  .cp-chip {{
    display: inline-block; background: rgba(255,255,255,0.13); color: #EAF4F5;
    border: 1px solid rgba(255,255,255,0.22); border-radius: 999px;
    padding: 0.2rem 0.7rem; font-size: 0.74rem; margin-right: 0.4rem;
    letter-spacing: 0.02em;
  }}

  .cp-kpi {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 0.95rem 1.05rem; height: 100%;
    box-shadow: 0 1px 2px rgba(10,37,64,0.04);
  }}
  .cp-kpi .cp-label {{
    color: {SLATE}; font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.3rem;
    line-height: 1.25; min-height: 2.1em;
  }}
  .cp-kpi .cp-value {{
    color: {NAVY}; font-weight: 680; line-height: 1.15; white-space: nowrap;
    font-size: clamp(1.02rem, 1.45vw, 1.6rem);
  }}
  .cp-kpi .cp-sub {{ color: {SLATE}; font-size: 0.78rem; margin-top: 0.3rem; }}
  .cp-kpi .cp-up {{ color: {TEAL}; font-weight: 600; }}
  .cp-kpi .cp-down {{ color: {CRIMSON}; font-weight: 600; }}

  .cp-panel {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-left: 3px solid {TEAL}; border-radius: 10px;
    padding: 1rem 1.15rem; margin-bottom: 0.9rem;
  }}
  .cp-panel h4 {{ margin: 0 0 0.45rem 0; font-size: 0.95rem; }}
  .cp-panel p {{ margin: 0; color: {NAVY}; font-size: 0.88rem; line-height: 1.5; }}

  .cp-note {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-left: 3px solid {AMBER};
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.8rem;
    color: {NAVY}; font-size: 0.85rem; line-height: 1.5;
  }}
  .cp-danger {{ border-left-color: {CRIMSON}; }}
  .cp-ok {{ border-left-color: {TEAL}; }}

  .cp-tag {{
    display: inline-block; border-radius: 6px; padding: 0.12rem 0.5rem;
    font-size: 0.7rem; font-weight: 650; letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .cp-tag-live {{ background: {TEAL_PALE}; color: {TEAL}; border: 1px solid {TEAL_LIGHT}; }}
  .cp-tag-offline {{ background: #EEF2F6; color: {SLATE}; border: 1px solid {BORDER}; }}
  .cp-tag-warn {{ background: #FDF3E3; color: {AMBER}; border: 1px solid #EBD6AE; }}
  .cp-tag-stop {{ background: #FCEDEE; color: {CRIMSON}; border: 1px solid #EFC9CD; }}

  .cp-foot {{
    color: {SLATE}; font-size: 0.78rem; border-top: 1px solid {BORDER};
    padding-top: 0.8rem; margin-top: 2rem; line-height: 1.55;
  }}

  section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] h2 {{ font-size: 1rem; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {SLATE}; }}
  .stTabs [aria-selected="true"] {{ color: {NAVY}; }}
  div[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.3rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_dataset(seed: int, days: int) -> pd.DataFrame:
    return load_dataset(seed=seed, days=days)


@st.cache_data(show_spinner=False)
def get_filtered(
    seed: int,
    days: int,
    start: Optional[date],
    end: Optional[date],
    channels: Tuple[str, ...],
    devices: Tuple[str, ...],
    service_lines: Tuple[str, ...],
) -> pd.DataFrame:
    frame = get_dataset(seed, days)
    return analytics.apply_filters(
        frame,
        date_range=(start, end) if start and end else None,
        channels=list(channels),
        devices=list(devices),
        service_lines=list(service_lines),
    )


@st.cache_data(show_spinner=False)
def get_evidence(
    seed: int,
    days: int,
    start: Optional[date],
    end: Optional[date],
    channels: Tuple[str, ...],
    devices: Tuple[str, ...],
    service_lines: Tuple[str, ...],
    capture_rate: float,
) -> Dict[str, Any]:
    frame = get_filtered(seed, days, start, end, channels, devices, service_lines)
    return analytics.build_evidence_pack(
        frame,
        filters_applied={
            "date_start": str(start) if start else None,
            "date_end": str(end) if end else None,
            "channels": list(channels),
            "devices": list(devices),
            "service_lines": list(service_lines),
        },
        capture_rate=capture_rate,
    )


# --------------------------------------------------------------------------
# Small UI primitives
# --------------------------------------------------------------------------

def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_class = {"up": "cp-up", "down": "cp-down"}.get(tone, "")
    sub_html = f'<div class="cp-sub {tone_class}">{sub}</div>' if sub else ""
    return (
        f'<div class="cp-kpi"><div class="cp-label">{label}</div>'
        f'<div class="cp-value">{value}</div>{sub_html}</div>'
    )


def render_kpis(cards: Sequence[str]) -> None:
    columns = st.columns(len(cards), gap="small")
    for column, html in zip(columns, cards):
        column.markdown(html, unsafe_allow_html=True)


def panel(title: str, body: str) -> None:
    st.markdown(
        f'<div class="cp-panel"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def note(body: str, kind: str = "") -> None:
    extra = {"danger": " cp-danger", "ok": " cp-ok"}.get(kind, "")
    st.markdown(f'<div class="cp-note{extra}">{body}</div>', unsafe_allow_html=True)


def base_layout(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=15, color=NAVY)) if title else None,
        margin=dict(l=10, r=10, t=46 if title else 18, b=10),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  color=NAVY, size=12),
        colorway=SEQUENCE,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER,
                        font=dict(color=NAVY, size=12)),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zerolinecolor="#EEF2F6")
    fig.update_yaxes(gridcolor="#EEF2F6", zerolinecolor="#EEF2F6")
    return fig


def masthead(subtitle: str, chips: Sequence[str]) -> None:
    chip_html = "".join(f'<span class="cp-chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="cp-masthead"><h1>CarePath AI</h1>'
        f'<p>{subtitle}</p>'
        f'<div class="cp-chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        f'<div class="cp-foot"><strong>{privacy.DISCLAIMER}</strong><br>'
        f'{privacy.NOT_COMPLIANCE_NOTICE}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state() -> None:
    defaults: Dict[str, Any] = {
        "page": PAGES[0],
        "copilot_history": [],
        "copilot_question": "",
        "experiments": [],
        "selected_design_index": 0,
        "last_response": None,
        "last_screening": None,
        "eval_results": None,
        "force_fallback": False,
        "model_name": resolve_model(),
        "temperature": 0.2,
        "margin_override": 280.0,
        "capture_rate": experiment.DEFAULT_CAPTURE_RATE,
        "horizon_days": 90,
        "implementation_cost": 0.0,
        "last_simulation": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar(full: pd.DataFrame) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            f'<div style="padding:0.2rem 0 0.9rem 0;">'
            f'<div style="font-size:1.12rem;font-weight:700;color:{NAVY};">'
            f'🩺 CarePath AI</div>'
            f'<div style="font-size:0.76rem;color:{SLATE};margin-top:0.15rem;">'
            f'Healthcare Conversion Intelligence</div></div>',
            unsafe_allow_html=True,
        )

        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")

        st.divider()
        st.markdown("##### Filters")

        min_date = pd.Timestamp(full["date"].min()).date()
        max_date = pd.Timestamp(full["date"].max()).date()
        picked = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            start, end = picked[0], picked[1]
        else:
            start, end = min_date, max_date

        channels = st.multiselect("Acquisition channel", CHANNELS, default=CHANNELS)
        devices = st.multiselect("Device", DEVICES, default=DEVICES)
        service_lines = st.multiselect("Service line", SERVICE_LINES,
                                       default=SERVICE_LINES)

        if st.button("Reset filters", width="stretch"):
            for key in ("Date range",):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.markdown("##### AI copilot status")
        api_key = resolve_api_key()
        if api_key and not st.session_state["force_fallback"]:
            st.markdown(
                '<span class="cp-tag cp-tag-live">Live model</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Key {mask_key(api_key)} · model `{st.session_state['model_name']}`")
        elif api_key:
            st.markdown(
                '<span class="cp-tag cp-tag-warn">Fallback forced</span>',
                unsafe_allow_html=True,
            )
            st.caption("A key is configured but offline mode is switched on.")
        else:
            st.markdown(
                '<span class="cp-tag cp-tag-offline">Offline fallback</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "No API key configured. Answers come from the deterministic "
                "analyst and nothing leaves this machine."
            )

        st.divider()
        meta = describe_dataset(full)
        st.markdown("##### Data provenance")
        st.caption(
            f"Synthetic · seed `{meta.seed}` · {meta.days} days · "
            f"{meta.rows:,} rows · signature `{meta.signature}`"
        )
        st.caption(privacy.DISCLAIMER)

    return {
        "page": page,
        "start": start,
        "end": end,
        "channels": tuple(channels),
        "devices": tuple(devices),
        "service_lines": tuple(service_lines),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Funnel(
            y=funnel["stage_label"],
            x=funnel["count"],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=[NAVY, NAVY_SOFT, TEAL, TEAL_LIGHT, "#7FB3B8"]),
            connector=dict(line=dict(color=BORDER, width=1)),
            hovertemplate="<b>%{y}</b><br>%{x:,} visitors<extra></extra>",
        )
    )
    return base_layout(fig, height=360, title="Patient booking funnel")


def dropoff_waterfall(funnel: pd.DataFrame) -> go.Figure:
    labels = ["Sessions"]
    values = [int(funnel.iloc[0]["count"])]
    measures = ["absolute"]
    for _, row in funnel.iloc[1:].iterrows():
        labels.append(f"Lost at<br>{row['stage_label'].lower()}")
        values.append(-int(row["drop_off"]))
        measures.append("relative")
    labels.append("Attended")
    values.append(int(funnel.iloc[-1]["count"]))
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            decreasing=dict(marker=dict(color="#C9D6E0")),
            increasing=dict(marker=dict(color=TEAL)),
            totals=dict(marker=dict(color=NAVY)),
            connector=dict(line=dict(color=BORDER)),
            hovertemplate="%{x}<br>%{y:,}<extra></extra>",
        )
    )
    return base_layout(fig, height=360, title="Where volume is lost")


def step_rate_chart(table: pd.DataFrame, dimension: str) -> go.Figure:
    fig = go.Figure()
    for i, (key, _, _, label) in enumerate(analytics.STEP_DEFINITIONS):
        fig.add_trace(
            go.Bar(
                name=label,
                x=table[dimension],
                y=(table[key] * 100).round(2),
                marker_color=SEQUENCE[i % len(SEQUENCE)],
                hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", yaxis_title="Step conversion (%)")
    return base_layout(
        fig, height=400,
        title=f"Step conversion by {analytics.DIMENSION_LABELS[dimension].lower()}",
    )


def channel_efficiency_chart(table: pd.DataFrame) -> go.Figure:
    paid = table[table["marketing_spend"] > 0]
    fig = go.Figure(
        go.Bar(
            x=paid["cost_per_attended"].round(2),
            y=paid["channel"],
            orientation="h",
            marker_color=[CRIMSON if v == paid["cost_per_attended"].max() else TEAL
                          for v in paid["cost_per_attended"]],
            text=[f"${v:,.2f}" for v in paid["cost_per_attended"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>$%{x:,.2f} per attended appointment<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Cost per attended appointment (USD)",
        xaxis=dict(range=[0, float(paid["cost_per_attended"].max()) * 1.25]
                   if not paid.empty else None),
    )
    return base_layout(fig, height=300, title="Paid channel efficiency")


def channel_scatter(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=(table["session_to_booking_rate"] * 100).round(2),
            y=table["sessions"],
            mode="markers+text",
            text=table["channel"],
            textposition="top center",
            textfont=dict(size=11, color=NAVY),
            marker=dict(
                size=(table["appointments_attended"] ** 0.5) * 1.1,
                color=SEQUENCE[: len(table)],
                line=dict(color="#FFFFFF", width=1.5),
                sizemode="diameter",
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Sessions: %{y:,}<br>"
                "Session to booking: %{x:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="Session to completed booking (%)",
        yaxis_title="Sessions",
    )
    return base_layout(
        fig, height=400,
        title="Volume against conversion quality (bubble = attended appointments)",
    )


def trend_chart(trend: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["date"], y=trend["sessions"], name="Sessions",
            mode="lines", line=dict(color=NAVY, width=2),
            hovertemplate="%{x|%d %b}<br>%{y:,} sessions<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend["date"], y=(trend["session_to_booking_rate"] * 100).round(2),
            name="Session to booking (%)", mode="lines", yaxis="y2",
            line=dict(color=TEAL, width=2.4),
            hovertemplate="%{x|%d %b}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Sessions"),
        yaxis2=dict(title="Conversion (%)", overlaying="y", side="right",
                    showgrid=False),
    )
    return base_layout(fig, height=360, title="Daily traffic and conversion")


def device_trend_chart(trend: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, device in enumerate(sorted(trend["device"].unique())):
        subset = trend[trend["device"] == device]
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=(subset["completion_rate"] * 100).round(2),
                name=device, mode="lines",
                line=dict(color=SEQUENCE[i], width=2.2),
                hovertemplate=f"<b>{device}</b><br>%{{x|%d %b}}<br>"
                              "%{y:.1f}% completed<extra></extra>",
            )
        )
    fig.update_layout(yaxis_title="Booking start to completion (%)")
    return base_layout(fig, height=340,
                       title="Booking completion rate by device, over time")


def opportunity_chart(table: pd.DataFrame) -> go.Figure:
    labels = [f"{r['segment']} · {r['step'].lower()}" for _, r in table.iterrows()]
    values = table["projected_margin_opportunity"]
    # The largest opportunity is drawn in navy; the rest in teal.
    colors = [NAVY] + [TEAL] * (len(table) - 1)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f} projected<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Projected contribution margin opportunity (USD)",
        yaxis=dict(autorange="reversed"),
        # Headroom so the longest bar's outside label is not clipped.
        xaxis=dict(range=[0, float(values.max()) * 1.2] if len(values) else None),
    )
    return base_layout(fig, height=360,
                       title="Largest gaps against a best-in-class peer segment")


def simulation_chart(result: experiment.SimulationResult) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Control", "Treatment"],
            y=[result.control_rate * 100, result.treatment_rate * 100],
            marker_color=[SLATE, TEAL],
            text=[f"{result.control_rate * 100:.2f}%",
                  f"{result.treatment_rate * 100:.2f}%"],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(yaxis_title="Primary metric conversion (%)")
    return base_layout(fig, height=300, title="Simulated readout")


# --------------------------------------------------------------------------
# Page 1 - Growth Dashboard
# --------------------------------------------------------------------------

def _delta_sub(metrics: Dict[str, Any], key: str, as_pct: bool = False,
               lower_is_better: bool = False) -> Tuple[str, str]:
    entry = (metrics or {}).get(key)
    if not entry:
        return "", ""
    change = float(entry.get("relative_change", 0.0))
    arrow = "▲" if change >= 0 else "▼"
    good = (change < 0) if lower_is_better else (change >= 0)
    tone = "up" if good else "down"
    return f"{arrow} {abs(change) * 100:.1f}% vs prior half", tone


def page_dashboard(df: pd.DataFrame, evidence: Dict[str, Any]) -> None:
    masthead(
        "Understand patient journeys. Discover growth opportunities. "
        "Take smarter action.",
        ["Synthetic data", "90-day window", "Deterministic analytics"],
    )

    if df.empty:
        note("No rows match the current filters. Widen the selection in the "
             "sidebar to see the funnel.", kind="danger")
        footer()
        return

    kpi = evidence["kpi_summary"]
    comparison = evidence.get("period_comparison", {}) or {}
    metrics = comparison.get("metrics", {}) if comparison.get("available") else {}

    sessions_sub, sessions_tone = _delta_sub(metrics, "sessions")
    booking_sub, booking_tone = _delta_sub(metrics, "session_to_booking_rate")
    attended_sub, attended_tone = _delta_sub(metrics, "appointments_attended")
    cpa_sub, cpa_tone = _delta_sub(metrics, "cost_per_attended", lower_is_better=True)

    render_kpis(
        [
            kpi_card("Website sessions", f"{kpi['sessions']:,}",
                     sessions_sub, sessions_tone),
            kpi_card("Completed bookings", f"{kpi['bookings_completed']:,}",
                     f"{fmt_pct(kpi['session_to_booking_rate'], 2)} of sessions"),
            kpi_card("Appointments attended", f"{kpi['appointments_attended']:,}",
                     attended_sub, attended_tone),
            kpi_card("Marketing spend", fmt_money(kpi["marketing_spend"]),
                     f"{kpi['spend_rows_missing']} row(s) unknown"),
            kpi_card("Cost per attended", fmt_money(kpi["cost_per_attended"], 2),
                     cpa_sub, cpa_tone),
            kpi_card("Contribution margin", fmt_money(kpi["contribution_margin"]),
                     "estimated, assumption-based"),
        ]
    )

    st.markdown("")
    tabs = st.tabs(
        ["Funnel", "Segments", "Channel economics", "Trends", "Opportunities"]
    )

    # ---- Funnel -----------------------------------------------------------
    with tabs[0]:
        funnel = analytics.funnel_counts(df)
        left, right = st.columns(2, gap="medium")
        left.plotly_chart(funnel_chart(funnel), width="stretch")
        right.plotly_chart(dropoff_waterfall(funnel), width="stretch")

        leak = evidence["largest_drop_off"]
        panel(
            "Largest single loss of volume",
            f"<strong>{leak['step_label']}</strong> loses "
            f"{int(leak['lost']):,} visitors "
            f"({fmt_pct(leak['drop_off_rate'])} of the stage above). "
            "Largest is not the same as most addressable - most sessions that "
            "never reach a provider search were never booking-intent sessions. "
            "The Opportunities tab ranks gaps against a comparable peer segment "
            "instead, which is the more useful view for deciding what to work on.",
        )

        display = funnel.copy()
        display["step_conversion"] = display["step_conversion"].map(
            lambda v: f"{v * 100:.1f}%")
        display["drop_off_rate"] = display["drop_off_rate"].map(
            lambda v: f"{v * 100:.1f}%")
        display["share_of_sessions"] = display["share_of_sessions"].map(
            lambda v: f"{v * 100:.1f}%")
        display["count"] = display["count"].map(lambda v: f"{v:,}")
        display["drop_off"] = display["drop_off"].map(lambda v: f"{v:,}")
        st.dataframe(
            display[["stage_label", "count", "step_conversion", "drop_off",
                     "drop_off_rate", "share_of_sessions"]].rename(
                columns={
                    "stage_label": "Stage", "count": "Volume",
                    "step_conversion": "Step conversion", "drop_off": "Lost here",
                    "drop_off_rate": "Drop-off rate",
                    "share_of_sessions": "Share of sessions",
                }
            ),
            width="stretch", hide_index=True,
        )

    # ---- Segments ---------------------------------------------------------
    with tabs[1]:
        dimension = st.selectbox(
            "Compare by",
            analytics.DIMENSIONS,
            format_func=lambda d: analytics.DIMENSION_LABELS[d],
            index=1,
        )
        table = analytics.step_rates_by(df, dimension)
        st.plotly_chart(step_rate_chart(table, dimension), width="stretch")

        if dimension == "device" and len(df["device"].unique()) > 1:
            test = evidence["mobile_vs_desktop_completion_test"]
            trend = analytics.daily_trend(df, "device")
            st.plotly_chart(device_trend_chart(trend), width="stretch")
            panel(
                "Mobile versus desktop booking completion",
                f"Mobile completes {fmt_pct(test['rate_a'])} of booking starts "
                f"against {fmt_pct(test['rate_b'])} on desktop, a gap of "
                f"{fmt_pct(abs(test['difference']))} absolute on "
                f"{int(test['n_a']):,} and {int(test['n_b']):,} booking starts. "
                f"Two-proportion z-test: z = {test['z_statistic']}, "
                f"p = {test['p_value']}, 95% CI on the difference "
                f"[{test['ci_low']:.3f}, {test['ci_high']:.3f}]. "
                "The gap is real in this data. Its <em>cause</em> is not "
                "established - device is correlated with channel, intent and "
                "time of day, none of which are controlled for here.",
            )

        show = table.copy()
        for column in ["search_rate", "start_rate", "completion_rate",
                       "attendance_rate", "session_to_booking_rate"]:
            show[column] = show[column].map(lambda v: f"{v * 100:.1f}%")
        for column in ["marketing_spend", "contribution_margin"]:
            show[column] = show[column].map(lambda v: f"${v:,.0f}")
        show["cost_per_attended"] = show["cost_per_attended"].map(
            lambda v: f"${v:,.2f}")
        st.dataframe(
            show[[dimension, "sessions", "booking_starts", "bookings_completed",
                  "appointments_attended", "search_rate", "start_rate",
                  "completion_rate", "attendance_rate", "session_to_booking_rate",
                  "marketing_spend", "cost_per_attended"]],
            width="stretch", hide_index=True,
        )

    # ---- Channel economics ------------------------------------------------
    with tabs[2]:
        channels = analytics.channel_economics(df)
        if channels.empty:
            note("No channel data under the current filters.")
        else:
            left, right = st.columns([1, 1], gap="medium")
            paid = channels[channels["marketing_spend"] > 0]
            if not paid.empty:
                left.plotly_chart(channel_efficiency_chart(channels),
                                  width="stretch")
            else:
                left.info("No paid channels in the current selection.")
            right.plotly_chart(channel_scatter(channels), width="stretch")

            if not paid.empty:
                worst = paid.loc[paid["cost_per_attended"].idxmax()]
                panel(
                    "Least efficient paid channel",
                    f"<strong>{worst['channel']}</strong> costs "
                    f"{fmt_money(worst['cost_per_attended'], 2)} per attended "
                    f"appointment on {fmt_money(worst['marketing_spend'])} of spend "
                    f"and converts {fmt_pct(worst['session_to_booking_rate'], 2)} of "
                    f"its sessions into bookings. It still returns "
                    f"{worst['return_on_spend']:.1f}x in estimated contribution "
                    "margin, so the useful question is whether its landing "
                    "experience can be fixed before its budget is cut.",
                )

            show = channels.copy()
            show["session_to_booking_rate"] = show["session_to_booking_rate"].map(
                lambda v: f"{v * 100:.2f}%")
            show["cost_per_attended"] = show["cost_per_attended"].map(
                lambda v: f"${v:,.2f}" if v > 0 else "unpaid")
            show["return_on_spend"] = show["return_on_spend"].map(
                lambda v: f"{v:,.1f}x" if v and v != float("inf") else "n/a")
            for column in ["marketing_spend", "contribution_margin"]:
                show[column] = show[column].map(lambda v: f"${v:,.0f}")
            st.dataframe(show, width="stretch", hide_index=True)

            for line in privacy.missing_data_notes(df):
                st.caption(f"Data note: {line}")

    # ---- Trends -----------------------------------------------------------
    with tabs[3]:
        trend = analytics.daily_trend(df)
        st.plotly_chart(trend_chart(trend), width="stretch")
        if evidence.get("period_comparison", {}).get("available"):
            comparison = evidence["period_comparison"]
            rows = []
            for metric, entry in comparison["metrics"].items():
                rows.append(
                    {
                        "Metric": metric.replace("_", " ").capitalize(),
                        "First half": entry["prior"],
                        "Second half": entry["recent"],
                        "Change": f"{entry['relative_change'] * 100:+.1f}%",
                    }
                )
            st.markdown(
                f"**Split at {comparison['split_date']}** · "
                f"{comparison['prior_days']} days before, "
                f"{comparison['recent_days']} days after"
            )
            st.dataframe(pd.DataFrame(rows), width="stretch",
                         hide_index=True)
            note(
                "A period split is a before-and-after comparison, not an "
                "experiment. Anything else that changed in the same weeks - "
                "budget, seasonality, competitor activity - is baked into these "
                "numbers.",
            )

    # ---- Opportunities ----------------------------------------------------
    with tabs[4]:
        capture = st.slider(
            "Share of each gap assumed to be captured",
            min_value=0.05, max_value=1.0,
            value=float(st.session_state["capture_rate"]), step=0.05,
            help="1.0 means full parity with the best-performing peer segment, "
                 "which is an upper bound rather than a plan.",
        )
        st.session_state["capture_rate"] = capture

        ranked = analytics.ranked_opportunities(df, capture_rate=capture, top_n=8)
        if ranked.empty:
            note("No segment gaps to size under the current filters.")
        else:
            display = ranked.rename(columns={"step_label": "step"})
            st.plotly_chart(
                opportunity_chart(
                    display.rename(columns={"opportunity_value":
                                            "projected_margin_opportunity"})
                ),
                width="stretch",
            )
            table = pd.DataFrame(
                {
                    "Dimension": ranked["dimension"].map(
                        analytics.DIMENSION_LABELS.get),
                    "Segment": ranked["segment"],
                    "Step": ranked["step_label"],
                    "Segment rate": (ranked["step_rate"] * 100).map(
                        lambda v: f"{v:.1f}%"),
                    "Benchmark": ranked["benchmark_segment"],
                    "Benchmark rate": (ranked["benchmark_rate"] * 100).map(
                        lambda v: f"{v:.1f}%"),
                    "Projected extra attended": ranked["recoverable_attended"].map(
                        lambda v: f"{v:,.0f}"),
                    "Projected margin": ranked["opportunity_value"].map(
                        lambda v: f"${v:,.0f}"),
                }
            )
            st.dataframe(table, width="stretch", hide_index=True)
            note(
                f"<strong>These are projections, not forecasts.</strong> Each row "
                f"assumes a lagging segment reaches {capture * 100:.0f}% of the way "
                "to a peer segment's observed rate, that the extra conversions "
                "behave like existing ones downstream, and that the contribution "
                "margin assumption holds. The rows overlap - fixing one moves the "
                "others - so they must not be summed.",
            )

    footer()


# --------------------------------------------------------------------------
# Page 2 - AI Conversion Copilot
# --------------------------------------------------------------------------

def render_answer(response: AnalystResponse) -> None:
    mode_tag = {
        "llm": '<span class="cp-tag cp-tag-live">Live model</span>',
        "fallback": '<span class="cp-tag cp-tag-offline">Deterministic fallback</span>',
        "error-fallback": '<span class="cp-tag cp-tag-warn">Fallback after model error</span>',
        "blocked": '<span class="cp-tag cp-tag-stop">Blocked</span>',
    }.get(response.mode, "")

    grounding = (
        '<span class="cp-tag cp-tag-live">All figures verified</span>'
        if response.grounded
        else f'<span class="cp-tag cp-tag-warn">'
             f'{len(response.unsupported_numbers)} figure(s) unverified</span>'
    )
    if response.mode == "blocked":
        grounding = ""

    meta = f"{response.latency_seconds:.2f}s"
    if response.model:
        meta += f" · {response.model}"
    if response.usage:
        meta += f" · {response.usage.get('total_tokens', '?')} tokens"

    st.markdown(
        f'<div style="margin-bottom:0.55rem;">{mode_tag} {grounding} '
        f'<span style="color:{SLATE};font-size:0.78rem;margin-left:0.4rem;">'
        f'{meta}</span></div>',
        unsafe_allow_html=True,
    )

    # A native bordered container, rather than a raw <div>: Streamlit closes
    # every markdown element it renders, so a hand-written wrapper div would
    # show up as an empty box instead of enclosing the answer.
    with st.container(border=True):
        st.markdown(response.answer)

    for warning in response.warnings:
        note(warning, kind="danger" if response.mode == "blocked" else "")


def page_copilot(df: pd.DataFrame, evidence: Dict[str, Any]) -> None:
    masthead(
        "Ask a business question. Every number in the answer is computed by the "
        "analytics layer, not generated by the model.",
        ["Evidence-grounded", "Aggregates only", "Works without an API key"],
    )

    if df.empty:
        note("No rows match the current filters, so there is nothing to analyse.",
             kind="danger")
        footer()
        return

    with st.expander("Copilot settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        st.session_state["model_name"] = c1.text_input(
            "Model", value=st.session_state["model_name"],
            help="Read from st.secrets or OPENAI_MODEL when set.",
        )
        st.session_state["temperature"] = c2.slider(
            "Temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.1,
            help="Lower values keep the interpretation close to the evidence.",
        )
        st.session_state["force_fallback"] = c3.toggle(
            "Force offline fallback",
            value=bool(st.session_state["force_fallback"]),
            help="Answer from the deterministic analyst even if a key is "
                 "configured. Nothing is transmitted.",
        )
        key = resolve_api_key()
        st.caption(
            f"API key: **{mask_key(key)}**. Keys are read from `st.secrets` or the "
            "`OPENAI_API_KEY` environment variable and are never written to source, "
            "logged, or shown in full."
        )

    analyst = AIAnalyst(
        model=st.session_state["model_name"],
        temperature=float(st.session_state["temperature"]),
        force_fallback=bool(st.session_state["force_fallback"]),
    )

    st.markdown("##### Try a question")
    columns = st.columns(len(SUGGESTED_QUESTIONS))
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        if column.button(question, width="stretch", key=f"sug-{question}"):
            st.session_state["copilot_question"] = question
            st.rerun()

    question = st.text_area(
        "Your question",
        key="copilot_question",
        height=90,
        placeholder="e.g. Why are mobile visitors converting less?",
    )

    ask_col, clear_col, _ = st.columns([1, 1, 4])
    asked = ask_col.button("Ask the copilot", type="primary",
                           width="stretch")
    if clear_col.button("Clear history", width="stretch"):
        st.session_state["copilot_history"] = []
        st.session_state["last_response"] = None
        st.rerun()

    if asked:
        screening = privacy.screen_question(question)
        st.session_state["last_screening"] = screening
        with st.spinner("Computing evidence and composing an answer..."):
            response = analyst.ask(question, evidence)
        st.session_state["last_response"] = response
        st.session_state["copilot_history"].insert(0, response)

    response = st.session_state.get("last_response")
    if response is None:
        panel(
            "How this copilot works",
            "Your question is screened for personal data, then an aggregated "
            "evidence pack is built by <code>analytics.py</code> from the rows "
            "currently selected in the sidebar. Only that pack is sent. The "
            "answer is then checked figure by figure against it, and anything "
            "that cannot be traced back is flagged. With no API key configured "
            "the whole flow still runs - a deterministic rule-based analyst "
            "answers from the same evidence and nothing leaves this machine.",
        )
        footer()
        return

    st.markdown("")
    render_answer(response)

    if response.mode != "blocked":
        st.markdown("")
        ev_tab, prompt_tab, data_tab = st.tabs(
            ["Model input preview", "Full prompt", "Supporting numbers"]
        )
        with ev_tab:
            st.caption(
                "This is the complete payload that was sent. It is rebuilt from "
                "the privacy allowlist, so nothing outside the allowlist can "
                "appear here."
            )
            if response.removed_fields:
                note(
                    "Fields dropped by the allowlist: "
                    + ", ".join(response.removed_fields[:10]),
                    kind="danger",
                )
            st.json(response.evidence_sent, expanded=False)
            st.download_button(
                "Download this payload",
                data=json.dumps(response.evidence_sent, indent=2, default=str),
                file_name="carepath_model_input.json",
                mime="application/json",
            )
        with prompt_tab:
            st.caption(
                "The system prompt forbids inventing figures, requires hypotheses "
                "to be labelled as unproven, and requires the answer's structure."
            )
            with st.expander("System prompt", expanded=False):
                st.code(__import__("ai_analyst").SYSTEM_PROMPT, language="text")
            st.code(response.prompt_preview[:6000] +
                    ("\n... truncated for display ..."
                     if len(response.prompt_preview) > 6000 else ""),
                    language="json")
        with data_tab:
            st.caption("The tables the evidence pack was built from.")
            for dimension in analytics.DIMENSIONS:
                st.markdown(f"**{analytics.DIMENSION_LABELS[dimension]}**")
                st.dataframe(
                    analytics.step_rates_by(df, dimension),
                    width="stretch", hide_index=True,
                )

        if response.unsupported_numbers:
            note(
                "<strong>Evidence validation</strong>: the figures "
                f"{', '.join(response.unsupported_numbers[:8])} could not be "
                "matched to any value in the evidence pack. On a live model this "
                "usually means the model derived or rounded something itself. "
                "Treat those numbers as unverified and check them against the "
                "dashboard.",
                kind="danger",
            )
        else:
            note(
                "<strong>Evidence validation passed</strong>: every figure in this "
                "answer matches a value in the evidence pack within a 2% rounding "
                "tolerance.",
                kind="ok",
            )

        if st.button("Take this to the Experiment Studio →", type="secondary"):
            st.session_state["page"] = "AI Experiment Studio"
            st.rerun()

    if len(st.session_state["copilot_history"]) > 1:
        with st.expander(
            f"Earlier questions ({len(st.session_state['copilot_history']) - 1})"
        ):
            for earlier in st.session_state["copilot_history"][1:]:
                st.markdown(f"**{earlier.question}** · _{earlier.mode_label}_")
                st.markdown(earlier.answer)
                st.divider()

    footer()


# --------------------------------------------------------------------------
# Page 3 - AI Experiment Studio
# --------------------------------------------------------------------------

STATUS_TAGS = {
    experiment.STATUS_DRAFT: '<span class="cp-tag cp-tag-offline">Draft</span>',
    experiment.STATUS_IN_REVIEW: '<span class="cp-tag cp-tag-warn">Pending review</span>',
    experiment.STATUS_APPROVED: '<span class="cp-tag cp-tag-live">Approved to run</span>',
    experiment.STATUS_REJECTED: '<span class="cp-tag cp-tag-stop">Rejected</span>',
}


def render_backlog() -> None:
    records: List[experiment.ExperimentRecord] = st.session_state["experiments"]
    if not records:
        st.caption(
            "Nothing in the backlog yet. Size an experiment above and add it."
        )
        return

    summary = experiment.portfolio_summary(records)
    render_kpis(
        [
            kpi_card("Experiments", str(summary["total"])),
            kpi_card("Approved", str(summary["approved"])),
            kpi_card("Pending review", str(summary["pending_review"])),
            kpi_card("Projected margin, approved only",
                     fmt_money(summary["projected_margin_approved"]),
                     "projection, not a forecast"),
        ]
    )
    note(summary["note"])

    for index, record in enumerate(records):
        header = f"{record.design.title} — {record.status}"
        # Streamlit reruns on every interaction, so an expander that defaults
        # closed would snap shut mid-review. Keep the newest entry and anything
        # awaiting a decision open.
        keep_open = (
            record.status == experiment.STATUS_IN_REVIEW
            or index == len(records) - 1
        )
        with st.expander(header, expanded=keep_open):
            st.markdown(STATUS_TAGS.get(record.status, ""), unsafe_allow_html=True)
            st.markdown(f"**Hypothesis.** {record.design.hypothesis}")
            st.markdown(f"**Proposed change.** {record.design.proposed_change}")
            st.markdown(
                f"**Primary metric.** {record.design.primary_metric} · "
                f"**Projected incremental margin.** "
                f"{fmt_money(record.projection.projected_incremental_margin)} "
                f"over {record.projection.horizon_days} days (projection)"
            )
            if record.reviewer:
                st.caption(f"Reviewer: {record.reviewer}")
            if record.review_notes:
                st.caption(f"Notes: {record.review_notes}")

            cols = st.columns([2, 1, 1, 1])
            reviewer = cols[0].text_input(
                "Reviewer name", value=record.reviewer,
                key=f"reviewer-{index}",
                placeholder="Required to approve or reject",
                label_visibility="collapsed",
            )
            if record.status == experiment.STATUS_DRAFT:
                if cols[1].button("Submit for review", key=f"submit-{index}",
                                  width="stretch"):
                    record.transition(experiment.STATUS_IN_REVIEW, reviewer)
                    st.rerun()
            if record.status == experiment.STATUS_IN_REVIEW:
                if cols[1].button("Approve", key=f"approve-{index}",
                                  type="primary", width="stretch"):
                    if record.transition(experiment.STATUS_APPROVED, reviewer,
                                         "Approved for a controlled test."):
                        st.rerun()
                    else:
                        st.warning("Enter a reviewer name before approving.")
                if cols[2].button("Reject", key=f"reject-{index}",
                                  width="stretch"):
                    if record.transition(experiment.STATUS_REJECTED, reviewer,
                                         "Rejected at review."):
                        st.rerun()
                    else:
                        st.warning("Enter a reviewer name before rejecting.")
            if record.status in (experiment.STATUS_APPROVED,
                                 experiment.STATUS_REJECTED):
                if cols[1].button("Return to draft", key=f"draft-{index}",
                                  width="stretch"):
                    record.transition(experiment.STATUS_DRAFT, reviewer)
                    st.rerun()

            cols[3].download_button(
                "Export JSON",
                data=json.dumps(record.as_dict(), indent=2, default=str),
                file_name=f"carepath_experiment_{index + 1}.json",
                mime="application/json",
                key=f"dl-{index}",
                width="stretch",
            )

            if record.history:
                st.caption("Audit trail")
                st.dataframe(pd.DataFrame(record.history),
                             width="stretch", hide_index=True)


def page_experiment_studio(df: pd.DataFrame, evidence: Dict[str, Any]) -> None:
    masthead(
        "Turn a recommendation into a measurable, powered, human-approved "
        "experiment.",
        ["Projections labelled", "Powered sample size", "Human approval gate"],
    )

    designs = experiment.recommend_experiments(evidence, max_experiments=5)
    if not designs:
        note("No measurable gap is large enough to design an experiment around "
             "under the current filters. Widen the selection in the sidebar.",
             kind="danger")
        footer()
        return

    st.markdown("##### 1. Choose a candidate")
    index = st.selectbox(
        "Candidate experiments, ranked by measured opportunity",
        range(len(designs)),
        format_func=lambda i: designs[i].title,
        index=min(st.session_state["selected_design_index"], len(designs) - 1),
    )
    st.session_state["selected_design_index"] = index
    design = designs[index]

    panel("Why this candidate exists", design.rationale)

    st.markdown("##### 2. Set your business assumptions")
    c1, c2, c3, c4 = st.columns(4)
    margin = c1.number_input(
        "Contribution margin per attended appointment (USD)",
        min_value=25.0, max_value=2000.0,
        value=float(st.session_state["margin_override"]), step=5.0,
    )
    capture = c2.slider(
        "Share of the measured gap captured", 0.05, 1.0,
        float(st.session_state["capture_rate"]), 0.05,
    )
    horizon = c3.number_input(
        "Projection horizon (days)", min_value=7, max_value=365,
        value=int(st.session_state["horizon_days"]), step=7,
    )
    cost = c4.number_input(
        "One-off implementation cost (USD)", min_value=0.0, max_value=5_000_000.0,
        value=float(st.session_state["implementation_cost"]), step=1000.0,
    )
    st.session_state.update(
        {"margin_override": margin, "capture_rate": capture,
         "horizon_days": horizon, "implementation_cost": cost}
    )

    c5, c6, c7 = st.columns(3)
    alpha = c5.select_slider("Significance level (alpha)",
                             options=[0.01, 0.05, 0.10], value=0.05)
    power = c6.select_slider("Statistical power",
                             options=[0.70, 0.80, 0.90, 0.95], value=0.80)
    allocation = c7.slider("Share of eligible traffic in the test", 0.1, 1.0, 1.0, 0.1)

    assumptions = experiment.BusinessAssumptions(
        margin_per_attended=margin,
        capture_rate=capture,
        horizon_days=int(horizon),
        implementation_cost=cost,
        alpha=float(alpha),
        power=float(power),
    )
    projection = experiment.project_impact(design, assumptions, allocation)

    st.markdown("##### 3. The experiment")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown(f"**Hypothesis.** {design.hypothesis}")
        st.markdown(f"**Proposed change.** {design.proposed_change}")
        st.markdown(
            f"**Target segment.** {analytics.DIMENSION_LABELS[design.dimension]} "
            f"= `{design.segment}` · "
            f"{design.eligible_units_in_window:,} eligible units over "
            f"{design.window_days} days"
        )
        st.markdown(f"**Primary metric.** {design.primary_metric}")
        st.markdown("**Guardrail metrics.**")
        for guardrail in design.guardrail_metrics:
            st.markdown(f"- {guardrail}")
    with right:
        st.markdown(
            f"**Design.** Randomised 50/50 split, "
            f"{allocation * 100:.0f}% of eligible traffic.\n\n"
            f"**Baseline.** {fmt_pct(projection.baseline_rate, 2)}\n\n"
            f"**Target at {capture * 100:.0f}% capture.** "
            f"{fmt_pct(projection.target_rate, 2)} "
            f"(a {fmt_pct(projection.relative_lift, 1)} relative lift)\n\n"
            f"**Sample size.** {projection.sample_size_per_arm:,} per arm "
            f"at alpha {alpha}, power {power}\n\n"
            f"**Estimated runtime.** {projection.estimated_runtime_days:,.1f} days "
            f"at current traffic"
        )

    st.markdown("##### 4. Projected financial opportunity")
    note(
        "<strong>Everything below is a projection under the assumptions you set "
        "above, not a forecast and not a result.</strong> No experiment has been "
        "run. The figures move directly with the capture-rate and margin inputs - "
        "try changing them to see how much of the headline number is assumption.",
    )
    render_kpis(
        [
            kpi_card("Projected extra conversions",
                     f"{projection.projected_incremental_conversions:,.0f}",
                     f"at the {design.step_label.lower()} step"),
            kpi_card("Projected extra attended appointments",
                     f"{projection.projected_incremental_attended:,.0f}",
                     "through observed downstream rates"),
            kpi_card("Projected contribution margin",
                     fmt_money(projection.projected_incremental_margin),
                     f"over {projection.horizon_days} days"),
            kpi_card("Projected net of cost",
                     fmt_money(projection.projected_net_margin),
                     f"after {fmt_money(projection.implementation_cost)} build cost"),
            kpi_card("Projected annualised",
                     fmt_money(projection.projected_annualised_margin),
                     "linear scaling, ignores seasonality"),
        ]
    )

    with st.expander("How this number was built, and what could break it",
                     expanded=False):
        st.markdown(
            f"```\n"
            f"{projection.eligible_units_per_day:,.0f} eligible units/day\n"
            f"  x {projection.horizon_days} days\n"
            f"  x {projection.absolute_lift:.4f} absolute lift\n"
            f"  = {projection.projected_incremental_conversions:,.0f} extra conversions\n"
            f"  x {design.downstream_to_attended:.4f} observed downstream conversion\n"
            f"  = {projection.projected_incremental_attended:,.0f} extra attended appointments\n"
            f"  x ${projection.margin_per_attended:,.0f} assumed margin each\n"
            f"  = ${projection.projected_incremental_margin:,.0f} projected contribution margin\n"
            f"```"
        )
        for caveat in projection.caveats:
            st.markdown(f"- {caveat}")

    st.markdown("##### 5. Simulate the readout")
    st.caption(
        "A synthetic A/B draw from an assumed true lift. This shows how the "
        "readout and its statistics would behave at your sample size. It is not "
        "evidence that the lift exists."
    )
    s1, s2, s3, s4 = st.columns([2, 2, 1, 1])
    true_lift = s1.slider(
        "Assumed true relative lift", 0.0, 0.60,
        float(min(projection.relative_lift, 0.60)), 0.01,
    )
    n_per_arm = s2.number_input(
        "Units per arm", min_value=100, max_value=500_000,
        value=int(max(projection.sample_size_per_arm, 100)), step=100,
    )
    seed = s3.number_input("Seed", min_value=0, max_value=9999, value=7, step=1)
    run = s4.button("Run simulation", width="stretch")

    if run:
        st.session_state["last_simulation"] = (
            design.title,
            experiment.simulate_experiment(
                design, true_lift, int(n_per_arm), alpha=float(alpha), seed=int(seed)
            ),
        )

    # Only show a stored result if it belongs to the candidate on screen,
    # so switching candidates never leaves a stale readout behind.
    stored = st.session_state.get("last_simulation")
    result = stored[1] if stored and stored[0] == design.title else None
    if result is not None:
        left, right = st.columns([1, 1], gap="medium")
        left.plotly_chart(simulation_chart(result), width="stretch")
        with right:
            verdict = (
                '<span class="cp-tag cp-tag-live">Significant</span>'
                if result.significant
                else '<span class="cp-tag cp-tag-warn">Not significant</span>'
            )
            st.markdown(verdict, unsafe_allow_html=True)
            st.markdown(
                f"**Observed lift.** {fmt_pct(result.observed_absolute_lift, 2)} "
                f"absolute ({fmt_pct(result.observed_relative_lift, 1)} relative)\n\n"
                f"**p-value.** {result.p_value:.5g} at alpha {alpha}\n\n"
                f"**95% CI on the difference.** "
                f"[{result.ci_low:.4f}, {result.ci_high:.4f}]\n\n"
                f"**Conversions.** {result.control_conversions:,} control vs "
                f"{result.treatment_conversions:,} treatment, "
                f"{result.n_per_arm:,} per arm"
            )
        for caveat in result.caveats:
            st.caption(f"• {caveat}")

    st.markdown("##### 6. Human approval")
    note(
        "CarePath AI never launches anything. An experiment stays in "
        "<strong>Draft</strong> until a named person submits it for review and a "
        "named reviewer approves it. The transition is recorded in an audit trail.",
    )
    if st.button("Add this experiment to the backlog", type="primary"):
        record = experiment.build_record(design, assumptions, allocation)
        st.session_state["experiments"].append(record)
        st.success(f"Added '{design.title}' to the backlog as a Draft.")

    st.markdown("##### Experiment backlog")
    render_backlog()
    footer()


# --------------------------------------------------------------------------
# Page 4 - AI Trust & Privacy
# --------------------------------------------------------------------------

def page_trust(df: pd.DataFrame, evidence: Dict[str, Any]) -> None:
    masthead(
        "What this product will and will not do with data, shown rather than "
        "claimed.",
        ["Synthetic only", "Allowlisted egress", "Human in the loop"],
    )

    note(
        f"<strong>{privacy.DISCLAIMER}</strong><br>{privacy.NOT_COMPLIANCE_NOTICE}",
        kind="danger",
    )

    tabs = st.tabs(
        ["Principles", "What gets sent", "Data quality", "Input screening",
         "AI evaluation"]
    )

    # ---- Principles -------------------------------------------------------
    with tabs[0]:
        for title, body in privacy.PRIVACY_PRINCIPLES:
            panel(title, body)

        st.markdown("##### Explicitly out of scope")
        for item in privacy.OUT_OF_SCOPE:
            st.markdown(f"- {item}")

        st.markdown("##### Current safeguard status")
        analyst = AIAnalyst(force_fallback=bool(st.session_state["force_fallback"]))
        sanitized = privacy.sanitize_evidence(evidence)
        checks = privacy.privacy_checklist(
            sanitization=sanitized,
            screening=st.session_state.get("last_screening"),
            api_configured=analyst.available,
        )
        st.dataframe(
            pd.DataFrame(checks).rename(
                columns={"check": "Safeguard", "status": "Status",
                         "detail": "Detail"}
            ),
            width="stretch", hide_index=True,
        )

    # ---- What gets sent ---------------------------------------------------
    with tabs[1]:
        sanitized = privacy.sanitize_evidence(evidence)
        render_kpis(
            [
                kpi_card("Payload size", f"{sanitized.size_bytes():,} bytes",
                         "aggregated metrics only"),
                kpi_card("Top-level sections", str(len(sanitized.payload))),
                kpi_card("Fields dropped by allowlist",
                         str(len(sanitized.removed)),
                         "0 means nothing unexpected appeared"),
                kpi_card("Row-level records sent", "0",
                         "structurally impossible"),
            ]
        )

        st.markdown("##### Model input preview")
        st.caption(
            "This is exactly what would be transmitted for the current filter "
            "selection. Nothing else is sent - no rows, no identifiers, no "
            "free text from the dataset."
        )
        st.json(sanitized.payload, expanded=False)
        st.download_button(
            "Download the payload",
            data=sanitized.as_json(),
            file_name="carepath_evidence_pack.json",
            mime="application/json",
        )

        st.markdown("##### The allowlist itself")
        st.caption(
            f"{len(privacy.ALLOWED_SECTIONS)} permitted sections and "
            f"{len(privacy.ALLOWED_FIELDS)} permitted field names. The payload is "
            "rebuilt from these names rather than filtered, so a new upstream "
            "field is excluded by default until it is deliberately added."
        )
        a, b = st.columns(2)
        a.markdown("**Permitted sections**")
        a.dataframe(pd.DataFrame({"section": sorted(privacy.ALLOWED_SECTIONS)}),
                    width="stretch", hide_index=True, height=260)
        b.markdown("**Permitted fields**")
        b.dataframe(pd.DataFrame({"field": sorted(privacy.ALLOWED_FIELDS)}),
                    width="stretch", hide_index=True, height=260)

        leaked = privacy.assert_payload_is_safe(sanitized.payload)
        if leaked:
            note(f"Final scan flagged: {', '.join(leaked)}", kind="danger")
        else:
            note(
                "Final backstop scan: the serialised payload contains no "
                "email addresses, phone numbers, national identifiers, record "
                "numbers or long numeric identifiers.",
                kind="ok",
            )

    # ---- Data quality -----------------------------------------------------
    with tabs[2]:
        st.markdown("##### Missing-data handling")
        for line in privacy.missing_data_notes(df):
            note(line)
        st.caption(
            "Missing spend is injected deliberately by the generator so that the "
            "handling path is exercised rather than assumed. Volumes are never "
            "imputed - a missing count would be reported, not filled in."
        )

        st.markdown("##### Column completeness")
        report = privacy.data_quality_report(df)
        st.dataframe(report, width="stretch", hide_index=True)

        st.markdown("##### Dataset provenance")
        meta = describe_dataset(get_dataset(DEFAULT_SEED, DEFAULT_DAYS))
        st.json(meta.as_dict())
        st.caption(
            "The signature is a hash of the funnel counts. It is identical on "
            "every machine, which is what makes every number in this app "
            "reproducible."
        )

    # ---- Input screening --------------------------------------------------
    with tabs[3]:
        st.markdown("##### Try the screener")
        st.caption(
            "Questions are screened before anything is sent. A match blocks the "
            "request outright rather than silently redacting it, so a user is "
            "never left thinking a truncated question was answered in full."
        )
        probe = st.text_input(
            "Test a question",
            value="Show me the patient name and date of birth for mobile bookings",
        )
        result = privacy.screen_question(probe)
        if result.allowed:
            note("This question would be <strong>allowed</strong>. No personal or "
                 "individual-level patterns were found.", kind="ok")
        else:
            note(
                "This question would be <strong>blocked</strong>.<br>"
                f"Detected: {', '.join(result.findings)}<br>{result.message}",
                kind="danger",
            )

        st.markdown("##### Patterns the screener looks for")
        st.dataframe(
            pd.DataFrame(
                [{"pattern": name, "regex": pattern}
                 for name, pattern in privacy.PII_PATTERNS]
            ),
            width="stretch", hide_index=True,
        )
        st.markdown("##### Phrases that indicate an individual-level request")
        st.dataframe(
            pd.DataFrame({"phrase": privacy.INDIVIDUAL_LEVEL_PHRASES}),
            width="stretch", hide_index=True, height=220,
        )

    # ---- AI evaluation ----------------------------------------------------
    with tabs[4]:
        st.markdown("##### Automated evaluation suite")
        st.caption(
            "Two suites: property checks on the deterministic analytics layer, "
            "and behavioural checks that put real questions to the configured "
            "analyst and grade the answers for grounding, structure, hedging, "
            "actionability, causal honesty and refusal of individual-level "
            "requests."
        )
        use_fallback = st.toggle(
            "Evaluate the offline fallback analyst",
            value=True,
            help="Switch off to evaluate the live model instead, which makes "
                 "one API call per case.",
        )
        if st.button("Run the evaluation suite", type="primary"):
            analyst = AIAnalyst(
                model=st.session_state["model_name"],
                temperature=float(st.session_state["temperature"]),
                force_fallback=use_fallback,
            )
            with st.spinner("Running evaluation..."):
                st.session_state["eval_results"] = evaluation.run_full_evaluation(
                    df, analyst, evidence
                )

        results = st.session_state.get("eval_results")
        if results:
            render_kpis(
                [
                    kpi_card("Checks run", str(results["total"])),
                    kpi_card("Passed", str(results["passed"])),
                    kpi_card("Failed", str(results["failed"])),
                    kpi_card("Pass rate", f"{results['pass_rate'] * 100:.0f}%",
                             f"analyst mode: {results['mode']}"),
                ]
            )
            combined = results["combined"]
            failed = combined[combined["result"] == "fail"]
            if not failed.empty:
                note(f"{len(failed)} check(s) failed. Details below.",
                     kind="danger")
            st.dataframe(combined, width="stretch", hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=combined.to_csv(index=False),
                file_name="carepath_evaluation_results.csv",
                mime="text/csv",
            )
        else:
            st.caption("No run yet. Press the button to evaluate the AI layer.")

    footer()


# --------------------------------------------------------------------------
# Page 5 - About
# --------------------------------------------------------------------------

IMPLEMENTED = [
    "Deterministic 90-day synthetic marketing dataset (5,400 rows, fixed seed)",
    "Funnel, segment, channel-economics, trend and period-comparison analytics",
    "Parity-gap opportunity sizing with an adjustable capture-rate assumption",
    "Two-proportion significance testing on segment differences",
    "AI copilot grounded in an aggregated evidence pack, with a full offline fallback",
    "Post-hoc evidence validation that flags figures the evidence cannot support",
    "Allowlist-based egress control, PII screening and a model input preview",
    "Experiment designer with powered sample size, runtime estimate and guardrails",
    "Financial projection engine with explicit, user-adjustable assumptions",
    "Synthetic A/B simulator with reproducible seeds",
    "Human approval workflow with an audit trail",
    "Automated evaluation suite runnable from inside the product",
]

PLANNED = [
    "Connectors to real analytics sources, with a privacy review before any is built",
    "Page-level and form-field event data, which is what the current funnel most lacks",
    "Causal methods - difference-in-differences, matched segments - instead of parity gaps",
    "Multi-touch attribution rather than the last-touch model baked into this dataset",
    "Experiment result ingestion, so projections can be scored against outcomes",
    "Sequential testing with proper alpha spending, to make early stopping safe",
    "Persistent storage and multi-user roles for the approval workflow",
    "A model-evaluation harness comparing providers on the same evidence pack",
]


def page_about() -> None:
    masthead(
        "An independent personal portfolio project in AI product management and "
        "applied analytics.",
        ["Personal project", "Synthetic data", "Prototype"],
    )

    st.markdown(
        """
### Why I built this

I wanted a project that forced me to work across the whole of an AI product, not
just the model call: generating a believable dataset, computing analysis that
holds up, deciding what an AI layer should and should not be trusted with,
designing an interface an executive could read, and building the guardrails that
make an AI recommendation safe to act on.

Healthcare marketing is a good setting for that, because the stakes make the
guardrails obvious. A marketing funnel is commercially meaningful, the data is
genuinely messy, and the moment you touch anything patient-adjacent the privacy
question stops being theoretical. It let me design for constraints rather than
bolt them on.

### The problem I framed

A hospital's website attracts a large volume of visitors who show real intent -
they search for a provider, they open the booking flow - and then disappear.
Marketing teams usually see the two ends of this: traffic at the top, booked
appointments at the bottom. What they often cannot see is *where* intent is lost,
*which segment* is losing it, and *what a fix would be worth*.

Analytics tools answer "what happened". Someone still has to turn that into
"what should we test, and is it worth the engineering time". That gap is where I
aimed this product.

### My hypothesis

A marketing team will act on an AI recommendation only if it can see the
arithmetic behind it. So the design bet here is a strict split: **deterministic
code computes every number, and the model is allowed only to interpret them.**
The model gets an aggregated evidence pack and is told, in the system prompt,
never to invent a figure. Afterwards the answer is checked figure by figure
against that pack and anything unverifiable is flagged in the interface.

That constraint costs some fluency. In exchange, every claim is auditable, the
product works with no API key at all, and a user who distrusts the AI can
re-derive its numbers from the dashboard.

### Who I designed it for

Hospital marketing managers, digital health growth teams, healthcare marketing
analysts and patient acquisition teams. Concretely: someone who owns a
conversion number, has to defend a budget, and has limited engineering time to
spend - so the product's real job is helping them choose *what not to build*.

### MVP scope and the decisions behind it

I deliberately kept four things out of the MVP.

**No real data connectors.** A connector would have made the demo more
impressive and the privacy story much worse. Synthetic data meant I could design
the egress controls honestly.

**No agentic actions.** The copilot proposes; a named human approves. Nothing
launches itself. For anything patient-adjacent, I think that is the correct
default.

**No causal claims.** The product measures gaps against comparable peer
segments. It never says a cause has been established, because this data cannot
establish one.

**No invented numbers.** This drove the whole architecture, including a fallback
analyst good enough to be the default experience.

### Technical architecture

```
data_generator.py   seeded synthetic dataset, deliberately planted signals
analytics.py        every metric, every rate, every projection
privacy.py          allowlist, PII screening, data-quality reporting
ai_analyst.py       prompt construction, model call, fallback, verification
experiment.py       design, power analysis, projection, simulation, approvals
evaluation.py       analytics invariants and AI behaviour checks
app.py              Streamlit presentation layer, no business logic
```

The dependency direction is one-way. `app.py` imports everything; nothing
imports `app.py`. The analytics layer has no knowledge of the AI layer, which is
what makes the AI layer replaceable.

### AI integration, and where it falls short

The model receives roughly ten kilobytes of aggregated metrics and a system
prompt that requires it to separate observation from hypothesis, state
uncertainty and recommend a specific action. Temperature is low by default.

Known limitations, stated plainly:

- The verifier matches figures numerically. It cannot catch a *correctly quoted
  number used to support a wrong conclusion*.
- The evidence pack is a fixed schema. The model cannot ask a follow-up question
  of the data, so a question outside the pack's coverage gets a partial answer.
- The fallback analyst is rule-based. It is reliable and reproducible, but it
  only recognises the intents I wrote handlers for.
- I have evaluated the copilot's behaviour, not its judgement. Whether its
  recommendations are *good* is not something this harness can tell you.

### Privacy approach

Synthetic data only; aggregates out, never records; an allowlist rebuilt
field-by-field rather than a blocklist; question screening that blocks rather
than redacts; the exact payload visible before and after every call; and a human
approval gate on anything actionable. This is privacy-aware design, which is not
the same as compliance - and this prototype makes no compliance claim.

### How I would measure whether it works

I have not run this with users, so these are the measures I would set up, not
results I have:

- **Decision quality** - what share of approved experiments actually ship, and
  what share of shipped ones beat their guardrails.
- **Projection calibration** - projected lift against measured lift, which is
  the number that would tell me whether the sizing model is honest.
- **Trust behaviour** - how often users open the evidence panel, and whether
  that changes once they do.
- **Time to a decision** - from question asked to experiment approved.
- **Grounding rate in production** - the share of answers passing verification
  with no flagged figures.
"""
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Implemented today")
        for item in IMPLEMENTED:
            st.markdown(f"- {item}")
    with right:
        st.markdown("### Planned, not built")
        for item in PLANNED:
            st.markdown(f"- {item}")

    note(
        "This separation is deliberate. Everything in the left column is "
        "running in this application right now and can be inspected. Nothing in "
        "the right column exists yet, and I have not claimed otherwise anywhere "
        "in this project.",
    )
    footer()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    init_state()
    full = get_dataset(DEFAULT_SEED, DEFAULT_DAYS)
    state = render_sidebar(full)

    df = get_filtered(
        DEFAULT_SEED, DEFAULT_DAYS, state["start"], state["end"],
        state["channels"], state["devices"], state["service_lines"],
    )
    evidence = get_evidence(
        DEFAULT_SEED, DEFAULT_DAYS, state["start"], state["end"],
        state["channels"], state["devices"], state["service_lines"],
        float(st.session_state["capture_rate"]),
    )

    page = state["page"]
    if page == "Growth Dashboard":
        page_dashboard(df, evidence)
    elif page == "AI Conversion Copilot":
        page_copilot(df, evidence)
    elif page == "AI Experiment Studio":
        page_experiment_studio(df, evidence)
    elif page == "AI Trust & Privacy":
        page_trust(df, evidence)
    else:
        page_about()


if __name__ == "__main__":
    main()
