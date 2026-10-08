"""R4动作差值监督核心：可微因果展开与严格分离的配对数据。"""
from __future__ import annotations

import torch
from torch import nn

from src.rl.r4_control import NominalCalibration, R4Limits, project_request
from src.rl.r4_dynamics import advance_history
from src.rl.r4_trajectory import anchor_delta


def pulse_rollout(model: nn.Module, history: torch.Tensor, valid: torch.Tensor,
                  correction: torch.Tensor, anchor: dict,
                  calibration: NominalCalibration, horizon: int = 8) -> torch.Tensor:
    """无未来命令/观测参数；动作、状态及名义队列梯度不断开。"""
    history, valid = history.clone(), valid.clone()
    powers = []
    for step in range(horizon):
        pulse = correction if step == 0 else torch.zeros_like(correction)
        request = project_request(history[:, -1, 21:42], anchor_delta(history[:, -1], anchor), pulse, R4Limits())
        prediction = model(history, valid, request.requested_delta_rad)
        if not bool(torch.isfinite(prediction).all()):
            raise RuntimeError('non-finite differentiable response')
        powers.append(prediction[:, 21])
        history, valid = advance_history(history, valid, request.requested_delta_rad,
            request.normalized_correction, prediction, calibration)
    return torch.stack(powers, dim=1)


def paired_losses(predicted: torch.Tensor, zero_predicted: torch.Tensor,
                  target: torch.Tensor, zero_target: torch.Tensor,
                  power_scale: torch.Tensor, delta_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not (predicted.shape == zero_predicted.shape == target.shape == zero_target.shape):
        raise ValueError('paired power shapes differ')
    if not bool((power_scale > 0).all()) or not bool((delta_scale > 0).all()):
        raise ValueError('positive training-only scales required')
    absolute = .5*(((predicted-target)/power_scale).square().mean() +
                    ((zero_predicted-zero_target)/power_scale).square().mean())
    delta = (((predicted.mean(1)-zero_predicted.mean(1))-
              (target.mean(1)-zero_target.mean(1)))/delta_scale).square().mean()
    return absolute, delta


class PairStore:
    """仅白名单数据；同一天气全部配对只属于一种拆分。CPU保存，CUDA计算。"""
    def __init__(self, records: list[dict], split: str):
        if split not in ('train', 'development') or not records:
            raise ValueError('nonempty explicit pair split required')
        self.split = split
        self.records = []
        rows = {k: [] for k in ('history', 'valid', 'correction', 'power', 'zero_power', 'weather')}
        self.metadata = []
        for item in records:
            if item['split'] != split or item['source'] != 'simulation_residual_proxy_not_holography':
                raise ValueError('pair source/split mismatch')
            r = {k: item[k] for k in ('history', 'valid', 'corrections', 'powers', 'zero_power',
                                      'weather', 'family', 'profile', 'probe')}
            b = len(r['weather']); a = len(r['corrections'])
            if (r['history'].shape != (b, 8, 79) or r['valid'].shape != (b, 8)
                    or r['powers'].shape != (a, b, 8) or r['zero_power'].shape != (b, 8)
                    or r['corrections'].shape != (a, 11)):
                raise ValueError('invalid pair record shape')
            for field in ('history', 'corrections', 'powers', 'zero_power'):
                if not bool(torch.isfinite(r[field]).all()):
                    raise ValueError('non-finite paired data')
            self.records.append(r)
            for j in range(a):
                for field, value in [('history', r['history']), ('valid', r['valid']),
                                     ('correction', r['corrections'][j].expand(b, -1)),
                                     ('power', r['powers'][j]), ('zero_power', r['zero_power']),
                                     ('weather', r['weather'])]:
                    rows[field].append(value)
                mode = int(r['corrections'][j].abs().argmax()); sign = int(r['corrections'][j, mode])
                self.metadata.extend(dict(weather_seed=int(w), family=r['family'], profile=r['profile'],
                    probe=r['probe'], mode=mode, sign=sign) for w in r['weather'])
        self.data = {k: torch.cat(v) for k, v in rows.items()}
        self.count = len(self.data['weather'])

    def delta_scale(self, device: torch.device, floor: float) -> torch.Tensor:
        if self.split != 'train':
            raise ValueError('development must never define training scale')
        # 与标签计算相同：先按float32时间平均，再以float64归约尺度。
        diff = self.data['power'].to(device).mean(1)-self.data['zero_power'].to(device).mean(1)
        return diff.double().square().mean().sqrt().clamp_min(floor).float().detach()

    def pool(self, seed: int) -> torch.Tensor:
        if self.split != 'train':
            raise ValueError('development cannot supply training pool')
        weather = self.data['weather']; unique = weather.unique(sorted=True)
        generator = torch.Generator().manual_seed(seed)
        drawn = unique[torch.randint(len(unique), (len(unique),), generator=generator)]
        return torch.cat([torch.nonzero(weather == w).flatten() for w in drawn])

    def sample(self, pool: torch.Tensor, batch: int, generator: torch.Generator,
               device: torch.device) -> dict[str, torch.Tensor]:
        if self.split != 'train':
            raise ValueError('development cannot supply training samples')
        index = pool[torch.randint(len(pool), (batch,), generator=generator)]
        return {k: self.data[k][index].to(device) for k in
                ('history', 'valid', 'correction', 'power', 'zero_power')}


def accuracy_gain(actual: torch.Tensor, candidate: torch.Tensor, comparator: torch.Tensor,
                  threshold: torch.Tensor, weather: torch.Tensor, family: torch.Tensor,
                  seed: int, replicates: int) -> dict:
    """唯一预声明的两集合配对差；整个天气按类别分层抽样。"""
    if not (actual.shape == candidate.shape == comparator.shape == threshold.shape == weather.shape == family.shape):
        raise ValueError('unaligned paired statistics')
    if any(not bool(torch.isfinite(x).all()) for x in (actual, candidate, comparator, threshold)):
        raise ValueError('non-finite paired statistics')
    pos, neg = actual > threshold, actual < -threshold
    unique, inverse = weather.unique(sorted=True, return_inverse=True)
    counts = torch.zeros(len(unique), 6, device=actual.device, dtype=torch.float64)
    counts.index_add_(0, inverse, torch.stack((pos, neg, pos & (candidate > 0), neg & (candidate < 0),
                                              pos & (comparator > 0), neg & (comparator < 0)), 1).double())
    total = counts.sum(0)
    if not bool((total[:2] > 0).all()):
        return dict(status='INSUFFICIENT_SIGNAL', difference=None, ci95=None)
    def gain(x: torch.Tensor) -> torch.Tensor:
        return .5*((x[..., 2]-x[..., 4])/x[..., 0] + (x[..., 3]-x[..., 5])/x[..., 1])
    generator = torch.Generator(device=actual.device).manual_seed(seed)
    draws = torch.zeros(replicates, 6, device=actual.device, dtype=torch.float64)
    for f in family.unique(sorted=True):
        members = weather[family == f].unique(sorted=True)
        if any(bool((family[weather == w] != f).any()) for w in members):
            raise ValueError('weather assigned to multiple families')
        indices = torch.searchsorted(unique, members)
        selection = torch.randint(len(indices), (replicates, len(indices)), device=actual.device, generator=generator)
        draws += counts[indices[selection]].sum(1)
    valid = (draws[:, :2] > 0).all(1)
    ci = torch.quantile(gain(draws[valid]), torch.tensor([.025, .975], device=actual.device,
                         dtype=torch.float64)).tolist() if valid.any() else None
    return dict(status='ANALYZED', difference=float(gain(total)), ci95=ci,
                invalid_replicates=int((~valid).sum()), weather=len(unique))
