"""在 CUDA 上调优并配对评价 S2 传统模态控制基线。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import resolve_device
from src.models.resunet_phase import ResUNetPhase
from src.simulation import (
    AdaptiveOpticsEnv,
    ResUNetModalController,
    S1EnvConfig,
    load_s1_config,
    make_controller,
)
from src.simulation.evaluation import (
    ControllerRollout,
    export_rollout_h5,
    paired_delta,
    run_controller_rollout,
    summarize_oracle_upper_bound,
    summarize_rollouts,
)


ControllerFactory = Callable[[], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/s2_baselines_v1.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只用每组第一个基础种子和最多50步，结果仅作诊断冒烟。",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="冻结全部控制器后，打开最终封存种子并加入动态ResUNet。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick and args.final:
        raise SystemExit("--quick and --final cannot be used together")
    experiment_path = _project_path(args.config)
    experiment = _load_yaml(experiment_path)
    environment_path = _project_path(experiment["environment_config"])
    env_config, requested_device = load_s1_config(environment_path)
    device = resolve_device(requested_device)

    evaluation = experiment["evaluation"]
    development_seeds = [int(seed) for seed in evaluation["development_seeds"]]
    test_split = "pilot"
    test_seeds = [int(seed) for seed in evaluation["pilot_test_seeds"]]
    steps = int(evaluation.get("steps", env_config.episode_length))
    if args.quick:
        test_split = "quick_smoke"
        development_seeds = [int(evaluation["smoke_development_seed"])]
        test_seeds = [int(evaluation["smoke_test_seed"])]
        steps = min(steps, 50)
    elif args.final:
        test_split = "final_sealed"
        test_seeds = [int(seed) for seed in evaluation["final_sealed_test_seeds"]]

    tuning_records: list[dict[str, Any]] = []
    leaky_parameters = _tune_leaky_integrator(
        experiment,
        env_config,
        device,
        development_seeds,
        steps,
        tuning_records,
    )
    predictor_parameters = _tune_linear_predictor(
        experiment,
        env_config,
        device,
        development_seeds,
        steps,
        tuning_records,
    )

    controller_parameters: dict[str, dict[str, Any]] = {
        "no_correction": {},
        "direct_projection": {},
        "leaky_integrator": leaky_parameters,
        "linear_predictor": predictor_parameters,
    }
    controller_factories: dict[str, ControllerFactory] = {
        name: (
            lambda name=name, parameters=parameters: make_controller(
                name,
                env_config.num_modes,
                env_config.modal_limit_rad,
                **parameters,
            )
        )
        for name, parameters in controller_parameters.items()
    }
    if args.final:
        resunet_factory, resunet_parameters = _load_dynamic_resunet_factory(
            experiment,
            env_config,
            device,
        )
        controller_factories["resunet_dynamic_memoryless"] = resunet_factory
        controller_parameters["resunet_dynamic_memoryless"] = resunet_parameters
    base_output_dir = _project_path(experiment["outputs"]["directory"])
    if args.quick:
        output_dir = base_output_dir / "quick_smoke"
    elif args.final:
        output_dir = base_output_dir / "final_comparison"
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    export_trajectories = bool(experiment["outputs"].get("export_test_trajectories", True))
    all_rollouts: dict[str, list[ControllerRollout]] = {}
    results: list[dict[str, Any]] = []

    for name, factory in controller_factories.items():
        parameters = controller_parameters[name]
        rollouts = _run_seed_set(
            env_config,
            device,
            test_seeds,
            steps,
            factory,
            include_oracle_upper_bound=name == "no_correction",
        )
        all_rollouts[name] = rollouts
        summary = summarize_rollouts(rollouts)
        summary["parameters"] = parameters
        if name != "no_correction":
            summary["paired_delta_vs_no_correction"] = {
                metric: paired_delta(rollouts, all_rollouts["no_correction"], metric)
                for metric in ("strehl", "power_in_bucket", "phase_rmse")
            }
        results.append(summary)

        if export_trajectories:
            for rollout in rollouts:
                trajectory_path = output_dir / "trajectories" / (
                    f"{name}_base_seed_{rollout.seed}.h5"
                )
                export_rollout_h5(
                    rollout,
                    trajectory_path,
                    env_config,
                    metadata={"experiment_config": str(experiment_path.relative_to(PROJECT_ROOT))},
                )

    oracle_summary = summarize_oracle_upper_bound(all_rollouts["no_correction"])
    best_controller = max(results, key=lambda item: item["power_in_bucket"]["mean"])
    gate = _gate_status(best_controller, final_comparison=args.final)
    summary_document = {
        "stage": "S2",
        "status": (
            "quick_smoke_only"
            if args.quick
            else "final_s2_comparison_completed"
            if args.final
            else "pilot_baselines_completed_resunet_pending"
        ),
        "scientific_claims_allowed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "experiment_config": str(experiment_path.relative_to(PROJECT_ROOT)),
        "experiment_config_sha256": _file_sha256(experiment_path),
        "environment_config": str(environment_path.relative_to(PROJECT_ROOT)),
        "environment": asdict(env_config),
        "git": _git_state(),
        "development_base_seeds": development_seeds,
        "test_split": test_split,
        "test_base_seeds": test_seeds,
        "episodes_per_base_seed": env_config.batch_size,
        "steps": steps,
        "tuning": tuning_records,
        "test_results": results,
        "oracle_modal_upper_bound": oracle_summary,
        "best_deployable_controller_by_power_in_bucket": best_controller["controller"],
        "s2_gate": gate,
        "resunet_baseline": (
            {
                "status": "completed_memoryless_dynamic_observer",
                "controller": "resunet_dynamic_memoryless",
                "checkpoint": experiment["resunet_dynamic"]["checkpoint"],
            }
            if args.final
            else {
                "status": "pending_in_pilot_comparison",
                "reason": "最终比较使用--final和独立封存回合。",
            }
        ),
        "note": (
            "这是纯仿真传统控制基线；理想模态上限忽略SLM约束与时延，"
            "不是可部署控制器，也不是RL结果。"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results_csv(output_dir / "controller_summary.csv", results + [oracle_summary])
    _write_tuning_csv(output_dir / "tuning_results.csv", tuning_records)
    print(json.dumps(summary_document, ensure_ascii=False, indent=2))


def _tune_leaky_integrator(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    device: torch.device,
    seeds: list[int],
    steps: int,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    values = experiment["tuning"]["leaky_integrator"]
    candidates = [
        {"gain": float(gain), "leak": float(leak)}
        for gain in values["gains"]
        for leak in values["leaks"]
    ]
    return _select_parameters(
        "leaky_integrator",
        candidates,
        config,
        device,
        seeds,
        steps,
        records,
    )


def _tune_linear_predictor(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    device: torch.device,
    seeds: list[int],
    steps: int,
    records: list[dict[str, Any]],
) -> dict[str, float | int]:
    candidates = [
        {
            "prediction_horizon": config.slm_delay_frames,
            "velocity_gain": float(gain),
        }
        for gain in experiment["tuning"]["linear_predictor"]["velocity_gains"]
    ]
    return _select_parameters(
        "linear_predictor",
        candidates,
        config,
        device,
        seeds,
        steps,
        records,
    )


def _select_parameters(
    controller_name: str,
    candidates: list[dict[str, Any]],
    config: S1EnvConfig,
    device: torch.device,
    seeds: list[int],
    steps: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    best_parameters: dict[str, Any] | None = None
    best_score = -float("inf")
    for parameters in candidates:
        rollouts = _run_seed_set(
            config,
            device,
            seeds,
            steps,
            lambda parameters=parameters: make_controller(
                controller_name,
                config.num_modes,
                config.modal_limit_rad,
                **parameters,
            ),
        )
        summary = summarize_rollouts(rollouts)
        score = float(summary["power_in_bucket"]["mean"])
        records.append(
            {
                "controller": controller_name,
                "parameters": parameters,
                "development_power_in_bucket": score,
                "development_strehl": float(summary["strehl"]["mean"]),
                "development_violation_fraction": float(
                    summary["violation_fraction"]["mean"]
                ),
            }
        )
        if score > best_score:
            best_score = score
            best_parameters = parameters
    if best_parameters is None:
        raise RuntimeError(f"no candidates were evaluated for {controller_name}")
    return best_parameters


def _run_seed_set(
    config: S1EnvConfig,
    device: torch.device,
    seeds: list[int],
    steps: int,
    factory: ControllerFactory,
    include_oracle_upper_bound: bool = False,
) -> list[ControllerRollout]:
    return [
        run_controller_rollout(
            config,
            device,
            factory(),
            seed,
            steps,
            include_oracle_upper_bound=include_oracle_upper_bound,
        )
        for seed in seeds
    ]


def _gate_status(best: dict[str, Any], final_comparison: bool) -> dict[str, Any]:
    paired = best.get("paired_delta_vs_no_correction", {})
    power = paired.get("power_in_bucket", {})
    strehl = paired.get("strehl", {})
    numerical_baseline_pass = (
        power.get("ci95_low", -float("inf")) > 0
        and strehl.get("ci95_low", -float("inf")) > 0
        and best["violation_fraction"]["mean"] <= 0.05
    )
    return {
        "traditional_modal_baseline_numerical_gate": (
            "pass" if numerical_baseline_pass else "not_passed"
        ),
        "full_s2_gate": (
            "pass" if numerical_baseline_pass and final_comparison else "in_progress"
        ),
        "blocking_item": (
            None
            if numerical_baseline_pass and final_comparison
            else "动态ResUNet与全部传统控制器尚未在最终封存回合共同评价"
        ),
    }


def _load_dynamic_resunet_factory(
    experiment: dict[str, Any],
    config: S1EnvConfig,
    device: torch.device,
) -> tuple[ControllerFactory, dict[str, Any]]:
    settings = experiment["resunet_dynamic"]
    result_path = _project_path(settings["output_directory"]) / "summary.json"
    if not result_path.exists():
        raise FileNotFoundError("dynamic ResUNet result is missing; run training first")
    training_result = json.loads(result_path.read_text(encoding="utf-8"))
    if training_result["gate"]["perception_gate"] != "PASS":
        raise RuntimeError("dynamic ResUNet perception gate did not pass")
    checkpoint_path = _project_path(settings["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResUNetPhase(
        in_channels=2,
        base_channels=int(settings["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    reference_environment = AdaptiveOpticsEnv(config, device)

    def factory() -> ResUNetModalController:
        return ResUNetModalController(
            config.num_modes,
            config.modal_limit_rad,
            model,
            reference_environment.basis,
            reference_environment.pupil,
            controller_name="resunet_dynamic_memoryless",
        )

    parameters = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "observation_kind": "ideal_pupil_intensity_and_wrapped_phase",
        "uses_history": False,
    }
    return factory, parameters


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    columns = [
        "controller",
        "episodes",
        "steps_per_episode",
        "mean_strehl",
        "strehl_ci95_low",
        "strehl_ci95_high",
        "mean_power_in_bucket",
        "power_in_bucket_ci95_low",
        "power_in_bucket_ci95_high",
        "mean_phase_rmse",
        "mean_violation_fraction",
        "mean_action_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "controller": item["controller"],
                    "episodes": item["episodes"],
                    "steps_per_episode": item["steps_per_episode"],
                    "mean_strehl": item["strehl"]["mean"],
                    "strehl_ci95_low": item["strehl"]["ci95_low"],
                    "strehl_ci95_high": item["strehl"]["ci95_high"],
                    "mean_power_in_bucket": item["power_in_bucket"]["mean"],
                    "power_in_bucket_ci95_low": item["power_in_bucket"]["ci95_low"],
                    "power_in_bucket_ci95_high": item["power_in_bucket"]["ci95_high"],
                    "mean_phase_rmse": item["phase_rmse"]["mean"],
                    "mean_violation_fraction": item.get("violation_fraction", {}).get("mean", ""),
                    "mean_action_latency_ms": item.get("action_latency_ms", {}).get("mean", ""),
                }
            )


def _write_tuning_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "controller",
                "parameters_json",
                "development_power_in_bucket",
                "development_strehl",
                "development_violation_fraction",
            ]
        )
        for item in records:
            writer.writerow(
                [
                    item["controller"],
                    json.dumps(item["parameters"], ensure_ascii=False, sort_keys=True),
                    item["development_power_in_bucket"],
                    item["development_strehl"],
                    item["development_violation_fraction"],
                ]
            )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


if __name__ == "__main__":
    main()
