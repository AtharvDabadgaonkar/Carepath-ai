"""Tests for the privacy and trust guardrails."""

from __future__ import annotations

import json

import pytest

import privacy


def test_evidence_pack_passes_allowlist_untouched(evidence):
    result = privacy.sanitize_evidence(evidence)
    assert result.clean
    assert result.removed == []
    assert set(result.payload) == set(evidence)


def test_allowlist_drops_unknown_top_level_section(evidence):
    tampered = dict(evidence)
    tampered["raw_visitor_records"] = [{"email": "a@b.com"}]
    result = privacy.sanitize_evidence(tampered)
    assert "raw_visitor_records" in result.removed
    assert "raw_visitor_records" not in result.payload
    assert not result.clean


def test_allowlist_drops_unknown_nested_field(evidence):
    tampered = json.loads(json.dumps(evidence, default=str))
    tampered["kpi_summary"]["patient_email"] = "someone@example.com"
    result = privacy.sanitize_evidence(tampered)
    assert "kpi_summary.patient_email" in result.removed
    assert "patient_email" not in result.payload["kpi_summary"]


def test_allowlist_strips_nested_field_inside_a_list(evidence):
    tampered = json.loads(json.dumps(evidence, default=str))
    tampered["by_channel"][0]["mrn"] = "MRN-88213"
    result = privacy.sanitize_evidence(tampered)
    assert any("mrn" in removed for removed in result.removed)
    assert "mrn" not in result.payload["by_channel"][0]


def test_allowlist_is_rebuild_not_filter(evidence):
    """A brand new upstream field must be excluded by default."""
    tampered = json.loads(json.dumps(evidence, default=str))
    tampered["kpi_summary"]["a_field_nobody_reviewed"] = 1
    result = privacy.sanitize_evidence(tampered)
    assert "a_field_nobody_reviewed" not in result.payload["kpi_summary"]


def test_sanitize_handles_empty_payload():
    result = privacy.sanitize_evidence({})
    assert result.payload == {}
    assert result.clean


@pytest.mark.parametrize(
    "text,expected",
    [
        ("contact me at jane.doe@hospital.org", "email address"),
        ("call 415-555-0199 please", "phone number"),
        ("ssn 123-45-6789", "social security number"),
        ("look up MRN 884412", "medical record number"),
        ("what is their DOB", "date of birth"),
        ("id 1234567890123", "long numeric identifier"),
        ("their member id is on file", "insurance or member id"),
    ],
)
def test_pii_patterns_detected(text, expected):
    assert expected in privacy.scan_text_for_pii(text)


def test_clean_business_text_has_no_findings():
    text = "Why did mobile booking completion fall to 44% in September 2025?"
    assert privacy.scan_text_for_pii(text) == []


@pytest.mark.parametrize("question", [
    "Why are mobile visitors converting less?",
    "Which marketing channel is underperforming?",
    "What should we optimize first to improve bookings?",
])
def test_legitimate_questions_are_allowed(question):
    assert privacy.screen_question(question).allowed


@pytest.mark.parametrize("question", [
    "List the patients who booked on mobile",
    "What is the patient name for the last booking?",
    "Email me at me@example.com with the results",
    "Who booked an appointment yesterday?",
])
def test_individual_level_questions_are_blocked(question):
    result = privacy.screen_question(question)
    assert not result.allowed
    assert result.findings
    assert "aggregated" in result.message


def test_empty_question_is_rejected():
    assert not privacy.screen_question("   ").allowed


def test_overlong_question_is_rejected():
    assert not privacy.screen_question("a" * 1300).allowed


def test_screening_result_serialises():
    payload = privacy.screen_question("Where are the largest losses?").as_dict()
    assert payload["allowed"] is True
    assert payload["findings"] == []


def test_payload_safety_backstop(evidence):
    clean = privacy.sanitize_evidence(evidence).payload
    assert privacy.assert_payload_is_safe(clean) == []


def test_payload_safety_backstop_catches_injected_pii(evidence):
    tampered = dict(evidence)
    tampered["kpi_summary"] = dict(evidence["kpi_summary"])
    tampered["kpi_summary"]["note"] = "reach me at leak@example.com"
    assert "email address" in privacy.assert_payload_is_safe(tampered)


def test_data_quality_report_covers_every_column(dataset):
    report = privacy.data_quality_report(dataset)
    assert len(report) == len(dataset.columns)
    spend = report[report["column"] == "marketing_spend"].iloc[0]
    assert spend["missing"] == int(dataset["marketing_spend"].isna().sum())


def test_data_quality_report_on_empty_frame(dataset):
    report = privacy.data_quality_report(dataset.iloc[0:0])
    assert report.empty


def test_missing_data_notes_mention_spend(dataset):
    notes = privacy.missing_data_notes(dataset)
    assert any("marketing spend" in note for note in notes)
    assert any("lower bound" in note for note in notes)


def test_missing_data_notes_on_empty_frame(dataset):
    notes = privacy.missing_data_notes(dataset.iloc[0:0])
    assert len(notes) == 1


def test_checklist_reports_offline_when_no_key():
    checks = privacy.privacy_checklist(api_configured=False)
    external = next(c for c in checks if c["check"] == "External model call")
    assert external["status"] == "offline"
    assert "nothing is transmitted" in external["detail"]


def test_checklist_flags_dropped_fields(evidence):
    tampered = dict(evidence)
    tampered["secret_section"] = {"x": 1}
    result = privacy.sanitize_evidence(tampered)
    checks = privacy.privacy_checklist(sanitization=result)
    allowlist_check = next(
        c for c in checks if "allowlist" in c["check"]
    )
    assert allowlist_check["status"] == "warn"


def test_disclaimer_text_is_exact():
    assert privacy.DISCLAIMER == (
        "Independent personal portfolio project. Uses synthetic demonstration "
        "data only. Not intended for clinical decisions or processing real "
        "patient information."
    )


def test_no_hipaa_compliance_claim():
    combined = " ".join(
        [privacy.DISCLAIMER, privacy.NOT_COMPLIANCE_NOTICE]
        + [body for _, body in privacy.PRIVACY_PRINCIPLES]
    ).lower()
    assert "hipaa compliant" not in combined.replace("not hipaa compliant", "")
