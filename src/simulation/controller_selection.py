"""纯仿真硬件误差条件下的传统控制器选择规则。"""

from __future__ import annotations

from typing import Any, Sequence


def select_robust_controller(
    controller_results: Sequence[dict[str, Any]],
    *,
    candidate_ids: Sequence[str],
    required_profile_ids: Sequence[str],
    incumbent_id: str,
    nominal_profile_id: str,
    max_nominal_power_drop_fraction: float,
) -> dict[str, Any]:
    """按预声明门槛和最差档位收益选择鲁棒传统控制器。

    先要求候选通过全部必过硬件档位，并且基准档位相对原控制器的
    桶内功率下降不超过容许值；再最大化最差档位的相对桶内功率增益。
    没有合格候选时返回 ``selected_controller=None``，不得放宽门槛。
    """
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if not required_profile_ids:
        raise ValueError("required_profile_ids must not be empty")
    if not 0 <= max_nominal_power_drop_fraction < 1:
        raise ValueError("max_nominal_power_drop_fraction must be in [0, 1)")

    by_controller: dict[str, dict[str, dict[str, Any]]] = {}
    for item in controller_results:
        controller = str(item["controller"])
        profile = str(item["profile"])
        if profile in by_controller.setdefault(controller, {}):
            raise ValueError(f"duplicate controller/profile result: {controller}/{profile}")
        by_controller[controller][profile] = item

    incumbent = by_controller.get(incumbent_id)
    if incumbent is None or nominal_profile_id not in incumbent:
        raise ValueError("incumbent nominal profile result is missing")
    incumbent_nominal_power = float(
        incumbent[nominal_profile_id]["power_in_bucket"]["mean"]
    )
    if incumbent_nominal_power <= 0:
        raise ValueError("incumbent nominal power must be positive")

    ranking: list[dict[str, Any]] = []
    required_set = set(map(str, required_profile_ids))
    for candidate_id in map(str, candidate_ids):
        profiles = by_controller.get(candidate_id)
        if profiles is None or not required_set.issubset(profiles):
            raise ValueError(f"required profile results are missing for {candidate_id}")
        if nominal_profile_id not in profiles:
            raise ValueError(f"nominal profile result is missing for {candidate_id}")
        required = [profiles[profile] for profile in required_profile_ids]
        profile_pass_count = sum(
            item["profile_gate"]["validation_gate"] == "PASS" for item in required
        )
        gains = [
            float(item["profile_gate"]["mean_relative_power_gain"])
            for item in required
        ]
        violations = [
            float(item["violation_fraction"]["mean"])
            for item in required
        ]
        nominal_power = float(profiles[nominal_profile_id]["power_in_bucket"]["mean"])
        nominal_change = nominal_power / incumbent_nominal_power - 1
        nominal_retention_pass = nominal_change >= -max_nominal_power_drop_fraction
        all_required_pass = profile_pass_count == len(required_profile_ids)
        eligible = all_required_pass and nominal_retention_pass
        ranking.append(
            {
                "controller": candidate_id,
                "eligible": eligible,
                "all_required_profiles_pass": all_required_pass,
                "profile_pass_count": profile_pass_count,
                "required_profile_count": len(required_profile_ids),
                "minimum_relative_power_gain": min(gains),
                "mean_relative_power_gain": sum(gains) / len(gains),
                "maximum_violation_fraction": max(violations),
                "nominal_power_change_vs_incumbent": nominal_change,
                "nominal_retention_pass": nominal_retention_pass,
            }
        )

    ranking.sort(
        key=lambda item: (
            bool(item["eligible"]),
            int(item["profile_pass_count"]),
            float(item["minimum_relative_power_gain"]),
            float(item["mean_relative_power_gain"]),
            -float(item["maximum_violation_fraction"]),
        ),
        reverse=True,
    )
    selected = next((item["controller"] for item in ranking if item["eligible"]), None)
    return {
        "validation_gate": "PASS" if selected is not None else "FAIL",
        "selected_controller": selected,
        "incumbent_controller": incumbent_id,
        "required_profile_ids": list(map(str, required_profile_ids)),
        "nominal_profile_id": nominal_profile_id,
        "max_nominal_power_drop_fraction": max_nominal_power_drop_fraction,
        "ranking": ranking,
    }


def validate_frozen_robust_controller(
    controller_results: Sequence[dict[str, Any]],
    *,
    frozen_controller_id: str,
    incumbent_id: str,
    required_profile_ids: Sequence[str],
    nominal_profile_id: str,
    max_nominal_power_drop_fraction: float,
    safety_fix_profile_ids: Sequence[str],
    max_violation_fraction: float,
    require_lower_violation_than_incumbent: bool,
    incremental_power_profile_ids: Sequence[str],
    min_incremental_power_delta_ci95_low: float,
) -> dict[str, Any]:
    """验证已冻结控制器，不允许利用未见结果重新选型。

    全部必过档位仍需相对不校正参照通过原门槛；此外显式检查开发阶段
    针对的慢响应安全问题和严重配准增益问题。函数只给出PASS/FAIL，
    不返回新的候选排名。
    """
    if not required_profile_ids:
        raise ValueError("required_profile_ids must not be empty")
    if not 0 <= max_nominal_power_drop_fraction < 1:
        raise ValueError("max_nominal_power_drop_fraction must be in [0, 1)")
    if not 0 <= max_violation_fraction <= 1:
        raise ValueError("max_violation_fraction must be in [0, 1]")

    by_controller: dict[str, dict[str, dict[str, Any]]] = {}
    for item in controller_results:
        controller = str(item["controller"])
        profile = str(item["profile"])
        if profile in by_controller.setdefault(controller, {}):
            raise ValueError(f"duplicate controller/profile result: {controller}/{profile}")
        by_controller[controller][profile] = item

    frozen = by_controller.get(str(frozen_controller_id))
    incumbent = by_controller.get(str(incumbent_id))
    if frozen is None:
        raise ValueError("frozen controller results are missing")
    if incumbent is None:
        raise ValueError("incumbent controller results are missing")

    required_ids = list(map(str, required_profile_ids))
    needed_ids = {
        nominal_profile_id,
        *required_ids,
        *map(str, safety_fix_profile_ids),
        *map(str, incremental_power_profile_ids),
    }
    for controller, profiles in (
        (frozen_controller_id, frozen),
        (incumbent_id, incumbent),
    ):
        missing = sorted(needed_ids - set(profiles))
        if missing:
            raise ValueError(f"profile results are missing for {controller}: {missing}")

    required = [frozen[profile] for profile in required_ids]
    profile_pass_count = sum(
        item["profile_gate"]["validation_gate"] == "PASS" for item in required
    )
    all_required_profiles_pass = profile_pass_count == len(required_ids)
    relative_gains = [
        float(item["profile_gate"]["mean_relative_power_gain"])
        for item in required
    ]
    violations = [
        float(item["violation_fraction"]["mean"])
        for item in required
    ]

    incumbent_nominal_power = float(
        incumbent[nominal_profile_id]["power_in_bucket"]["mean"]
    )
    if incumbent_nominal_power <= 0:
        raise ValueError("incumbent nominal power must be positive")
    frozen_nominal_power = float(
        frozen[nominal_profile_id]["power_in_bucket"]["mean"]
    )
    nominal_change = frozen_nominal_power / incumbent_nominal_power - 1
    nominal_retention_pass = nominal_change >= -max_nominal_power_drop_fraction

    safety_checks: list[dict[str, Any]] = []
    for profile in map(str, safety_fix_profile_ids):
        frozen_violation = float(frozen[profile]["violation_fraction"]["mean"])
        incumbent_violation = float(
            incumbent[profile]["violation_fraction"]["mean"]
        )
        candidate_safe = frozen_violation <= max_violation_fraction
        lower_than_incumbent = frozen_violation < incumbent_violation
        passed = candidate_safe and (
            lower_than_incumbent or not require_lower_violation_than_incumbent
        )
        safety_checks.append(
            {
                "profile": profile,
                "frozen_controller_violation_fraction": frozen_violation,
                "incumbent_violation_fraction": incumbent_violation,
                "candidate_safe": candidate_safe,
                "lower_than_incumbent": lower_than_incumbent,
                "validation_gate": "PASS" if passed else "FAIL",
            }
        )

    incremental_power_checks: list[dict[str, Any]] = []
    for profile in map(str, incremental_power_profile_ids):
        interval = frozen[profile]["paired_delta_vs_incumbent"]["power_in_bucket"]
        ci95_low = float(interval["ci95_low"])
        passed = ci95_low > min_incremental_power_delta_ci95_low
        incremental_power_checks.append(
            {
                "profile": profile,
                "mean": float(interval["mean"]),
                "ci95_low": ci95_low,
                "ci95_high": float(interval["ci95_high"]),
                "validation_gate": "PASS" if passed else "FAIL",
            }
        )

    safety_fixes_pass = all(
        item["validation_gate"] == "PASS" for item in safety_checks
    )
    incremental_power_pass = all(
        item["validation_gate"] == "PASS" for item in incremental_power_checks
    )
    passed = (
        all_required_profiles_pass
        and nominal_retention_pass
        and safety_fixes_pass
        and incremental_power_pass
    )
    return {
        "validation_gate": "PASS" if passed else "FAIL",
        "frozen_controller": str(frozen_controller_id),
        "incumbent_controller": str(incumbent_id),
        "controller_selection_allowed": False,
        "retuning_allowed": False,
        "required_profile_ids": required_ids,
        "all_required_profiles_pass": all_required_profiles_pass,
        "profile_pass_count": profile_pass_count,
        "required_profile_count": len(required_ids),
        "minimum_relative_power_gain": min(relative_gains),
        "mean_relative_power_gain": sum(relative_gains) / len(relative_gains),
        "maximum_violation_fraction": max(violations),
        "nominal_power_change_vs_incumbent": nominal_change,
        "nominal_retention_pass": nominal_retention_pass,
        "safety_fixes_pass": safety_fixes_pass,
        "safety_checks": safety_checks,
        "incremental_power_pass": incremental_power_pass,
        "incremental_power_checks": incremental_power_checks,
        "thresholds": {
            "max_nominal_power_drop_fraction": max_nominal_power_drop_fraction,
            "max_violation_fraction": max_violation_fraction,
            "require_lower_violation_than_incumbent": (
                require_lower_violation_than_incumbent
            ),
            "min_incremental_power_delta_ci95_low": (
                min_incremental_power_delta_ci95_low
            ),
        },
    }
