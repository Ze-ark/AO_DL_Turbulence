"""S4纯仿真鲁棒传统控制器选择规则测试。"""

import pytest

from src.simulation.controller_selection import (
    select_robust_controller,
    validate_frozen_robust_controller,
)


def _record(
    controller: str,
    profile: str,
    *,
    gate: str,
    gain: float,
    power: float,
    violation: float,
) -> dict:
    return {
        "controller": controller,
        "profile": profile,
        "power_in_bucket": {"mean": power},
        "violation_fraction": {"mean": violation},
        "profile_gate": {
            "validation_gate": gate,
            "mean_relative_power_gain": gain,
        },
    }


def test_selection_prefers_best_worst_profile_gain_among_eligible_candidates():
    results = [
        _record("frozen", "nominal", gate="PASS", gain=0.20, power=0.70, violation=0.01),
        _record("a", "nominal", gate="PASS", gain=0.19, power=0.69, violation=0.01),
        _record("a", "stress", gate="PASS", gain=0.12, power=0.66, violation=0.03),
        _record("b", "nominal", gate="PASS", gain=0.18, power=0.69, violation=0.01),
        _record("b", "stress", gate="PASS", gain=0.15, power=0.67, violation=0.04),
    ]

    selection = select_robust_controller(
        results,
        candidate_ids=["a", "b"],
        required_profile_ids=["nominal", "stress"],
        incumbent_id="frozen",
        nominal_profile_id="nominal",
        max_nominal_power_drop_fraction=0.03,
    )

    assert selection["validation_gate"] == "PASS"
    assert selection["selected_controller"] == "b"


def test_selection_fails_closed_when_no_candidate_passes_every_profile():
    results = [
        _record("frozen", "nominal", gate="PASS", gain=0.20, power=0.70, violation=0.01),
        _record("a", "nominal", gate="PASS", gain=0.19, power=0.69, violation=0.01),
        _record("a", "stress", gate="FAIL", gain=0.09, power=0.64, violation=0.06),
    ]

    selection = select_robust_controller(
        results,
        candidate_ids=["a"],
        required_profile_ids=["nominal", "stress"],
        incumbent_id="frozen",
        nominal_profile_id="nominal",
        max_nominal_power_drop_fraction=0.03,
    )

    assert selection["validation_gate"] == "FAIL"
    assert selection["selected_controller"] is None


def test_selection_rejects_excessive_nominal_degradation():
    results = [
        _record("frozen", "nominal", gate="PASS", gain=0.20, power=0.70, violation=0.01),
        _record("a", "nominal", gate="PASS", gain=0.11, power=0.60, violation=0.01),
        _record("a", "stress", gate="PASS", gain=0.16, power=0.67, violation=0.02),
    ]

    selection = select_robust_controller(
        results,
        candidate_ids=["a"],
        required_profile_ids=["nominal", "stress"],
        incumbent_id="frozen",
        nominal_profile_id="nominal",
        max_nominal_power_drop_fraction=0.03,
    )

    assert selection["validation_gate"] == "FAIL"
    assert selection["ranking"][0]["nominal_retention_pass"] is False


def test_selection_validates_nominal_drop_range():
    with pytest.raises(ValueError, match="max_nominal_power_drop_fraction"):
        select_robust_controller(
            [],
            candidate_ids=["a"],
            required_profile_ids=["nominal"],
            incumbent_id="frozen",
            nominal_profile_id="nominal",
            max_nominal_power_drop_fraction=1,
        )


def _validation_record(
    controller: str,
    profile: str,
    *,
    power: float,
    gain: float,
    violation: float,
    incremental_ci95_low: float = 0.01,
) -> dict:
    return {
        "controller": controller,
        "profile": profile,
        "power_in_bucket": {"mean": power},
        "violation_fraction": {"mean": violation},
        "profile_gate": {
            "validation_gate": "PASS",
            "mean_relative_power_gain": gain,
        },
        "paired_delta_vs_incumbent": {
            "power_in_bucket": {
                "mean": 0.02,
                "ci95_low": incremental_ci95_low,
                "ci95_high": 0.03,
            }
        },
    }


def _frozen_validation_results() -> list[dict]:
    return [
        _validation_record(
            "incumbent", "nominal", power=0.70, gain=0.20, violation=0.02
        ),
        _validation_record(
            "incumbent", "settling", power=0.69, gain=0.18, violation=0.08
        ),
        _validation_record(
            "incumbent", "registration", power=0.64, gain=0.11, violation=0.03
        ),
        _validation_record(
            "frozen", "nominal", power=0.69, gain=0.19, violation=0.01
        ),
        _validation_record(
            "frozen", "settling", power=0.68, gain=0.17, violation=0.01
        ),
        _validation_record(
            "frozen",
            "registration",
            power=0.68,
            gain=0.17,
            violation=0.01,
            incremental_ci95_low=0.02,
        ),
    ]


def _validate_frozen(results: list[dict]) -> dict:
    return validate_frozen_robust_controller(
        results,
        frozen_controller_id="frozen",
        incumbent_id="incumbent",
        required_profile_ids=["nominal", "settling", "registration"],
        nominal_profile_id="nominal",
        max_nominal_power_drop_fraction=0.03,
        safety_fix_profile_ids=["settling"],
        max_violation_fraction=0.05,
        require_lower_violation_than_incumbent=True,
        incremental_power_profile_ids=["registration"],
        min_incremental_power_delta_ci95_low=0.0,
    )


def test_frozen_validation_passes_without_reselecting_controller():
    gate = _validate_frozen(_frozen_validation_results())

    assert gate["validation_gate"] == "PASS"
    assert gate["frozen_controller"] == "frozen"
    assert gate["controller_selection_allowed"] is False
    assert gate["safety_fixes_pass"] is True
    assert gate["incremental_power_pass"] is True


def test_frozen_validation_fails_when_slow_response_is_not_safer():
    results = _frozen_validation_results()
    frozen_settling = next(
        item
        for item in results
        if item["controller"] == "frozen" and item["profile"] == "settling"
    )
    frozen_settling["violation_fraction"]["mean"] = 0.09

    gate = _validate_frozen(results)

    assert gate["validation_gate"] == "FAIL"
    assert gate["safety_fixes_pass"] is False


def test_frozen_validation_fails_when_registration_increment_is_uncertain():
    results = _frozen_validation_results()
    frozen_registration = next(
        item
        for item in results
        if item["controller"] == "frozen" and item["profile"] == "registration"
    )
    frozen_registration["paired_delta_vs_incumbent"]["power_in_bucket"][
        "ci95_low"
    ] = -0.001

    gate = _validate_frozen(results)

    assert gate["validation_gate"] == "FAIL"
    assert gate["incremental_power_pass"] is False
