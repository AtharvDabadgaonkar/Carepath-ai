"""Tests for the evaluation harness itself."""

from __future__ import annotations

import pytest

import evaluation
from ai_analyst import AIAnalyst, AnalystResponse


def test_analytics_invariants_all_pass(dataset):
    results = evaluation.run_analytics_evaluation(dataset)
    failed = results[results["result"] == "fail"]
    assert failed.empty, failed.to_dict(orient="records")
    assert len(results) >= 7


def test_ai_suite_passes_on_the_fallback_analyst(offline_analyst, evidence):
    results, responses = evaluation.run_ai_evaluation(offline_analyst, evidence)
    failed = results[results["result"] == "fail"]
    assert failed.empty, failed.to_dict(orient="records")
    assert len(responses) == len(evaluation.DEFAULT_CASES)


def test_full_evaluation_reports_a_pass_rate(dataset, offline_analyst, evidence):
    results = evaluation.run_full_evaluation(dataset, offline_analyst, evidence)
    assert results["total"] > 25
    assert results["failed"] == 0
    assert results["pass_rate"] == 1.0
    assert results["mode"] == "fallback"


def test_pii_case_must_be_blocked(offline_analyst, evidence):
    results, responses = evaluation.run_ai_evaluation(offline_analyst, evidence)
    assert responses["pii-refusal"].mode == "blocked"
    safety = results[results["category"] == "safety"]
    assert (safety["result"] == "pass").all()


# --------------------------------------------------- the checks catch failures

def _response(text: str) -> AnalystResponse:
    return AnalystResponse(question="q", answer=text, mode="llm")


def test_grounding_check_catches_an_invented_number():
    passed, detail = evaluation._check_grounded(
        _response("We lost 987,654 bookings."), {"kpi_summary": {"sessions": 1000}}
    )
    assert not passed
    assert "987,654" in detail


def test_structure_check_catches_a_missing_section():
    passed, detail = evaluation._check_structured(
        _response("**What the data shows** - something happened."), {}
    )
    assert not passed
    assert "hypothesis section" in detail


def test_hedging_check_catches_an_overconfident_answer():
    passed, _ = evaluation._check_hedged(
        _response("Mobile is broken. Fix the form. Done."), {}
    )
    assert not passed


def test_actionability_check_catches_a_non_answer():
    passed, _ = evaluation._check_actionable(
        _response("Conversion is lower on small screens."), {}
    )
    assert not passed


def test_overclaim_check_catches_a_causal_claim():
    passed, detail = evaluation._check_no_overclaim(
        _response("This proves the form is the cause."), {}
    )
    assert not passed
    assert "this proves" in detail


def test_checks_pass_on_a_well_formed_answer(evidence):
    text = (
        "**What the data shows**\n- Something, possibly.\n"
        "**Possible explanations**\n- A hypothesis, not proven; it may be wrong "
        "and we cannot tell from this correlation.\n"
        "**What I would test first**\n- Run an experiment.\n"
        "**Confidence and caveats**\n- Uncertain; assumption-heavy."
    )
    response = _response(text)
    for name in ("grounded", "structured", "hedged", "actionable", "no_overclaim"):
        passed, detail = evaluation.CHECK_FUNCTIONS[name](response, evidence)
        assert passed, f"{name} failed: {detail}"


# ------------------------------------------------------------- failure routing

def test_a_blocked_legitimate_question_is_recorded_as_a_failure(evidence):
    class AlwaysBlocks(AIAnalyst):
        def ask(self, question, evidence, screen=True):
            return AnalystResponse(question=question, answer="no", mode="blocked")

    results, _ = evaluation.run_ai_evaluation(
        AlwaysBlocks(api_key="", force_fallback=True),
        evidence,
        cases=[evaluation.EvalCase("probe", "Where are the largest losses?")],
        check_determinism=False,
    )
    assert (results["result"] == "fail").any()


def test_an_unblocked_pii_request_is_recorded_as_a_failure(evidence):
    class NeverBlocks(AIAnalyst):
        def ask(self, question, evidence, screen=True):
            return AnalystResponse(question=question, answer="here you go",
                                   mode="fallback")

    results, _ = evaluation.run_ai_evaluation(
        NeverBlocks(api_key="", force_fallback=True),
        evidence,
        cases=[c for c in evaluation.DEFAULT_CASES if c.expect_blocked],
        check_determinism=False,
    )
    assert (results["result"] == "fail").all()


def test_determinism_check_is_included_by_default(offline_analyst, evidence):
    results, _ = evaluation.run_ai_evaluation(offline_analyst, evidence)
    assert "reproducibility" in set(results["category"])
