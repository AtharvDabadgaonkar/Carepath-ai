"""Tests for the AI copilot: grounding, fallback, failure handling, screening."""

from __future__ import annotations

import pytest

import ai_analyst
from ai_analyst import AIAnalyst, SUGGESTED_QUESTIONS


# ---------------------------------------------------------------- key handling

def test_resolve_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert ai_analyst.resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert ai_analyst.resolve_api_key() == "env-key"


def test_resolve_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ai_analyst.resolve_api_key() is None


def test_resolve_model_default(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert ai_analyst.resolve_model() == ai_analyst.DEFAULT_MODEL


def test_mask_key_never_reveals_the_whole_key():
    masked = ai_analyst.mask_key("sk-abcdefghijklmnop")
    assert "abcdefghijklmnop" not in masked
    assert masked.startswith("sk-")
    assert ai_analyst.mask_key(None) == "not configured"


def test_no_api_key_is_present_in_source():
    """A hard-coded key must never appear in the module source."""
    import pathlib

    for name in ("ai_analyst.py", "app.py", "privacy.py", "analytics.py"):
        text = pathlib.Path(name).read_text()
        assert "sk-" not in text.replace('"sk-..."', ""), f"possible key in {name}"


# ------------------------------------------------------------ number extraction

@pytest.mark.parametrize(
    "text,expected",
    [
        ("44.0%", 44.0),
        ("$4.0M", 4_000_000.0),
        ("12.5k visits", 12_500.0),
        ("61,994 booking starts", 61_994.0),
    ],
)
def test_extract_numbers_normalises_suffixes(text, expected):
    assert ai_analyst.extract_numbers(text)[0][1] == pytest.approx(expected)


def test_suffix_must_be_attached_to_the_number():
    """'61,994 mobile' must not read the 'm' of 'mobile' as 'million'."""
    found = dict(ai_analyst.extract_numbers("61,994 mobile booking starts"))
    assert found["61,994"] == 61_994.0


# -------------------------------------------------------------- verification

def test_verify_numbers_accepts_evidence_values():
    evidence = {"kpi_summary": {"sessions": 501189, "rate": 0.4404}}
    assert ai_analyst.verify_numbers("501,189 sessions at 44.0%", evidence) == []


def test_verify_numbers_allows_rounding_within_tolerance():
    evidence = {"kpi_summary": {"margin": 4_015_534.74}}
    assert ai_analyst.verify_numbers("about $4.0M", evidence) == []


def test_verify_numbers_flags_invented_figures():
    evidence = {"kpi_summary": {"sessions": 501189}}
    flagged = ai_analyst.verify_numbers("we lost 87,432 bookings", evidence)
    assert "87,432" in flagged


def test_verify_numbers_ignores_numbers_from_the_question():
    evidence = {"kpi_summary": {"sessions": 100000}}
    flagged = ai_analyst.verify_numbers(
        "Over the last 37 days...", evidence, question="What happened in 37 days?"
    )
    assert flagged == []


def test_verify_numbers_ignores_small_integers():
    evidence = {"kpi_summary": {"sessions": 100000}}
    assert ai_analyst.verify_numbers("there are 3 hypotheses", evidence) == []


# ------------------------------------------------------------- intent routing

@pytest.mark.parametrize(
    "question,intent",
    [
        ("Why are mobile visitors converting less?", "device"),
        ("Where are the largest booking losses?", "dropoff"),
        ("Which marketing channel is underperforming?", "channel"),
        ("What should we optimize first?", "priority"),
        ("Which experiment offers the greatest potential impact?", "experiment"),
        ("Tell me about the business", "overview"),
    ],
)
def test_intent_classification(question, intent):
    assert ai_analyst.classify_intent(question) == intent


# ------------------------------------------------------------------- fallback

def test_fallback_mode_when_no_key(offline_analyst, evidence):
    response = offline_analyst.ask("Where are the largest booking losses?", evidence)
    assert response.mode == "fallback"
    assert response.model is None
    assert "deterministic" in " ".join(response.warnings).lower()


@pytest.mark.parametrize("question", SUGGESTED_QUESTIONS)
def test_every_suggested_question_is_fully_grounded(offline_analyst, evidence, question):
    response = offline_analyst.ask(question, evidence)
    assert response.mode == "fallback"
    assert response.unsupported_numbers == [], response.unsupported_numbers
    assert response.grounded


@pytest.mark.parametrize("question", SUGGESTED_QUESTIONS)
def test_every_answer_has_the_promised_sections(offline_analyst, evidence, question):
    text = offline_analyst.ask(question, evidence).answer.lower()
    assert "what the data shows" in text
    assert "possible explanation" in text
    assert "test first" in text
    assert "caveat" in text or "confidence" in text


def test_fallback_is_deterministic(evidence):
    first = ai_analyst.deterministic_answer("why is mobile worse?", evidence)
    second = ai_analyst.deterministic_answer("why is mobile worse?", evidence)
    assert first == second


def test_fallback_handles_an_empty_evidence_pack(offline_analyst):
    response = offline_analyst.ask("What should we optimize first?", {})
    assert response.mode == "fallback"
    assert response.answer


# -------------------------------------------------------------- screening path

def test_pii_question_is_blocked_before_anything_is_sent(offline_analyst, evidence):
    response = offline_analyst.ask(
        "Give me the patient name and DOB for every mobile booking", evidence
    )
    assert response.mode == "blocked"
    assert response.evidence_sent == {}
    assert response.prompt_preview == ""


def test_screening_can_be_bypassed_for_internal_calls(offline_analyst, evidence):
    response = offline_analyst.ask("where do we lose bookings", evidence, screen=False)
    assert response.mode == "fallback"


# ----------------------------------------------------------------- live model

class _FakeUsage:
    prompt_tokens, completion_tokens, total_tokens = 2000, 300, 2300


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def test_llm_path_is_used_when_a_key_is_configured(monkeypatch, evidence):
    analyst = AIAnalyst(api_key="test-key", model="fake-model")
    assert analyst.available

    def fake_call(messages):
        assert messages[0]["role"] == "system"
        assert "EVIDENCE" in messages[1]["content"]
        return "**What the data shows**\n- Nothing numeric here.", {
            "total_tokens": 2300
        }

    monkeypatch.setattr(analyst, "_call_model", fake_call)
    response = analyst.ask("Why is mobile worse?", evidence)
    assert response.mode == "llm"
    assert response.model == "fake-model"
    assert response.usage["total_tokens"] == 2300


def test_llm_hallucinated_numbers_are_flagged(monkeypatch, evidence):
    analyst = AIAnalyst(api_key="test-key")
    monkeypatch.setattr(
        analyst, "_call_model",
        lambda messages: ("Mobile lost exactly 987,654 bookings.", {}),
    )
    response = analyst.ask("Why is mobile worse?", evidence)
    assert response.mode == "llm"
    assert "987,654" in response.unsupported_numbers
    assert not response.grounded
    assert any("could not be matched" in w for w in response.warnings)


def test_model_failure_falls_back_and_surfaces_the_error(monkeypatch, evidence):
    analyst = AIAnalyst(api_key="test-key")

    def boom(messages):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(analyst, "_call_model", boom)
    response = analyst.ask("Where are the largest booking losses?", evidence)
    assert response.mode == "error-fallback"
    assert "connection refused" in " ".join(response.warnings)
    assert response.answer, "the user must still get an answer"


def test_force_fallback_prevents_any_call(monkeypatch, evidence):
    analyst = AIAnalyst(api_key="test-key", force_fallback=True)

    def must_not_run(messages):
        raise AssertionError("the model must not be called in forced fallback")

    monkeypatch.setattr(analyst, "_call_model", must_not_run)
    response = analyst.ask("Why is mobile worse?", evidence)
    assert response.mode == "fallback"
    assert not analyst.available
    assert analyst.api_configured


def test_evidence_is_sanitised_before_sending(monkeypatch, evidence):
    analyst = AIAnalyst(api_key="test-key")
    captured = {}

    def capture(messages):
        captured["prompt"] = messages[1]["content"]
        return "ok", {}

    monkeypatch.setattr(analyst, "_call_model", capture)
    tampered = dict(evidence)
    tampered["raw_records"] = [{"email": "leak@example.com"}]
    response = analyst.ask("Why is mobile worse?", tampered)
    assert "leak@example.com" not in captured["prompt"]
    assert "raw_records" in response.removed_fields


def test_system_prompt_states_the_hard_rules():
    prompt = ai_analyst.SYSTEM_PROMPT.lower()
    assert "never state a number that is not present in evidence" in prompt
    assert "hypothesis" in prompt
    assert "projection" in prompt


def test_status_reports_mode_without_leaking_the_key():
    status = AIAnalyst(api_key="sk-supersecretvalue").status()
    assert status["mode"] == "llm"
    assert "supersecret" not in status["key"]
