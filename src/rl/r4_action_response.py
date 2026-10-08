"""R4受限脉冲的物理分支、模型分支及按天气聚类的方向检查。"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch

from src.rl.r4_control import R4Limits, NominalCalibration, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_observation import PowerMeasurement, simulation_residual_proxy
from src.rl.r4_trajectory import anchor_delta


def pulse(batch: int, mode: int, sign: float, step: int, device: torch.device) -> torch.Tensor:
    result = torch.zeros(batch, 11, device=device)
    if step == 0 and sign:
        if mode not in range(11) or sign not in (-1., 1.): raise ValueError("invalid diagnostic pulse")
        result[:, mode] = sign
    return result


@torch.no_grad()
def physical_step(env, interface, sensor, profile, correction: torch.Tensor, anchor: dict) -> dict:
    t = interface.step
    command = interface.issue(anchor_delta(interface.snapshot().features[:, -1], anchor), correction, step=t)
    raw, _, terminated, truncated, info = env.step(command.requested_delta_rad)
    residual = simulation_residual_proxy(raw, generator=sensor, noise_std_rad=profile.observation_noise_std_rad)
    interface.observe_next(residual, step=t+1, power=PowerMeasurement(info["measured_power_in_bucket"],t,t+1))
    if not torch.allclose(command.requested_modal_rad, info["requested_modal"], atol=1e-6,rtol=0):
        raise RuntimeError("response diagnostic request alignment failed")
    if bool(terminated.any()) or bool(truncated.any()):
        raise RuntimeError("diagnostic branch reached episode boundary")
    return dict(power=info["measured_power_in_bucket"], requested_delta=command.requested_delta_rad,
        requested_modal=command.requested_modal_rad, applied_modal=info["applied_modal"],
        audit_power=info["reward_power_in_bucket"], audit_strehl=info["reward_strehl"],
        audit_violation=info["violation_fraction"], audit_phase_rmse=info["reward_phase_rmse"])


@torch.no_grad()
def physical_branch(env, interface, sensor, profile, anchor: dict, mode: int, sign: float,
                    horizon: int, on_step: Callable, measurement_seed: int | None = None) -> dict:
    # 分支复制只在实验驱动器中发生；模型不接收env或其中的任何真值。
    branch_env, branch_interface, branch_sensor = deepcopy((env, interface, sensor))
    if measurement_seed is not None:
        branch_env.measurement_generator.manual_seed(measurement_seed)
    values = []
    for k in range(horizon):
        values.append(physical_step(branch_env,branch_interface,branch_sensor,profile,
            pulse(env.config.batch_size,mode,sign,k,env.device),anchor))
        on_step()
    tensors = {name: torch.stack([v[name] for v in values],dim=1) for name in values[0]}
    if any(not bool(torch.isfinite(x).all()) for x in tensors.values()):
        raise RuntimeError("non-finite physical diagnostic")
    return tensors


@torch.no_grad()
def model_branch(model, history: torch.Tensor, valid: torch.Tensor, anchor: dict,
                 calibration: NominalCalibration, mode: int, sign: float, horizon: int,
                 limits: R4Limits = R4Limits()) -> dict:
    history, valid = history.clone(), valid.clone()
    powers, commands = [], []
    for k in range(horizon):
        correction = pulse(len(history),mode,sign,k,history.device)
        action = project_request(history[:,-1,21:42],anchor_delta(history[:,-1],anchor),correction,limits)
        prediction = model(history,valid,action.requested_delta_rad)
        if not bool(torch.isfinite(prediction).all()): raise RuntimeError("non-finite model response")
        powers.append(prediction[:,21]);commands.append(action.requested_delta_rad)
        # 禁止使用物理分支的未来观测或未来基座命令。
        history,valid = advance_history(history,valid,action.requested_delta_rad,
            action.normalized_correction,prediction,calibration)
    return dict(power=torch.stack(powers,dim=1), requested_delta=torch.stack(commands,dim=1))


def noise_threshold(differences: torch.Tensor, multiplier: float, floor: float) -> float:
    if differences.numel()<2 or not bool(torch.isfinite(differences).all()):
        raise ValueError("insufficient finite training repeat measurements")
    return max(floor, multiplier*float(differences.double().std(unbiased=True)))


def direction_statistics(actual: torch.Tensor, predicted: torch.Tensor, thresholds: torch.Tensor,
                         weather: torch.Tensor, family: torch.Tensor, *, seed: int,
                         replicates: int, min_weather: int = 50,
                         accuracy_min: float = .6, lower_exclusive: float = .5) -> dict:
    """主检验为单个预声明集合；天气内所有档位/时刻/模态一起重采样。"""
    if not (actual.shape==predicted.shape==thresholds.shape==weather.shape==family.shape):
        raise ValueError("direction vectors must align")
    if any(not bool(torch.isfinite(x).all()) for x in (actual,predicted,thresholds)):
        raise ValueError("non-finite direction data")
    mask = actual.abs()>thresholds
    pos,neg = (actual>0)&mask,(actual<0)&mask
    tp,tn = pos&(predicted>0),neg&(predicted<0)
    unique,inverse = weather.unique(sorted=True,return_inverse=True)
    counts = torch.zeros(len(unique),4,device=actual.device,dtype=torch.float64)
    counts.index_add_(0,inverse,torch.stack((tp,pos,tn,neg),dim=1).double())
    counts_sum=counts.sum(0)
    def accuracy(c):return .5*(c[...,0]/c[...,1]+c[...,2]/c[...,3])
    identifiable=int(((counts[:,1]+counts[:,3])>0).sum())
    base = dict(total_pairs=len(actual),identifiable_pairs=int(mask.sum()),identifiable_weather=identifiable,
        positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),
        predicted_ties=int((mask&(predicted==0)).sum()),bootstrap_replicates=replicates)
    if counts_sum[1]==0 or counts_sum[3]==0:
        return dict(base,status="INSUFFICIENT_SIGNAL",balanced_accuracy=None,ci95=None)
    # 每个天气只能属于一个家族。
    strata=[]
    for f in family.unique():
        w=weather[family==f].unique()
        if any(bool((family[weather==x]!=f).any()) for x in w):raise ValueError("weather spans families")
        strata.append(torch.searchsorted(unique,w))
    generator=torch.Generator(device=actual.device).manual_seed(seed)
    totals=torch.zeros(replicates,4,device=actual.device,dtype=torch.float64)
    for indices in strata:
        draws=torch.randint(len(indices),(replicates,len(indices)),device=actual.device,generator=generator)
        totals+=counts[indices[draws]].sum(1)
    valid=(totals[:,1]>0)&(totals[:,3]>0)
    intervals=torch.quantile(accuracy(totals[valid]),torch.tensor([.025,.975],device=actual.device,dtype=torch.float64)) if valid.any() else None
    point=float(accuracy(counts_sum))
    ci=intervals.tolist() if intervals is not None else None
    passed=identifiable>=min_weather and bool(valid.all()) and ci is not None and point>=accuracy_min and ci[0]>lower_exclusive
    status="ACTION_DIRECTION_PASS" if passed else ("INSUFFICIENT_SIGNAL" if identifiable<min_weather else "ACTION_DIRECTION_FAIL")
    return dict(base,status=status,balanced_accuracy=point,ci95=ci,invalid_bootstrap_replicates=int((~valid).sum()))
