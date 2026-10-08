"""G2-D12-B 小型确定性 CPU 单元测试，不产生正式科学性能。"""
from copy import deepcopy

import pytest
import torch

from scripts import train_s4_r5_g2_d12_candidate_scorer as scorer


def config():
    return scorer.pairing.source._load_yaml(scorer.pairing.source._project_path(scorer.CONFIG))


def rows_fixture(quick=False):
    rows = []
    source = scorer.pairing.source
    for condition in source.CONDITIONS:
        for wi, seed in enumerate(source.stream_manifest(quick)["weather_bases"]):
            for step in ([5] if quick else [25, 75, 150]):
                for member in range(1 if quick else 3):
                    for family in source.FAMILIES:
                        for slot in range(6):
                            rows.append(dict(hardware_condition=condition, weather_seed=seed,
                                probe_step=step, member=member, family=family, slot=slot,
                                sample_index=len(rows), weather_fold=None if quick else wi // 2,
                                candidate_order=list(source.CANDIDATES), metadata_not_model_input=True))
    return rows


def inputs_fixture():
    torch.manual_seed(4)
    history = torch.zeros(6, 8, 79)
    history[..., :21] = torch.randn(6, 8, 21) * .01
    history[..., 74] = torch.rand(6, 8)
    history[..., 75] = torch.arange(8)
    history[..., 76:78] = torch.arange(8)[None, :, None] - 1
    history[..., 78] = 1
    return {"history": history, "valid": torch.ones(6, 8, dtype=torch.bool),
            "command": torch.randn(6, 11)}


def test_frozen_budget_and_configuration():
    cfg = config(); scorer.contract(cfg)
    assert cfg["training"]["updates"] * 4 * len(cfg["training"]["seeds"]) * len(cfg["models"]) == 24000
    assert cfg["quick"]["updates"] * 2 == 4
    assert cfg["boundary"]["held_out_evaluation"] is False


@pytest.mark.parametrize("field,value", [("models", ["oracle"]), ("runtime", {"device": "cpu"}),
    ("split", {"unit": "random_frame"}), ("purpose", "closed_loop"),
    ("boundary", {"confirmation_access": True}), ("training", {"early_stopping": True}),
    ("dataset_config", "other.yaml")])
def test_contract_blocks_scope_or_data_changes(field, value):
    cfg = config(); cfg[field] = value
    with pytest.raises(ValueError, match="合同"):
        scorer.contract(cfg)


def test_full_weather_split_complete_and_disjoint():
    rows = rows_fixture(); splits = scorer.weather_splits(rows, quick=False)
    assert len(rows) == 2592 and len(splits) == 4
    all_held = []
    for split in splits:
        assert len(split["train_indices"]) == 1944 and len(split["held_out_indices"]) == 648
        assert set(split["train_weather"]).isdisjoint(split["held_out_weather"])
        assert set(split["train_indices"]).isdisjoint(split["held_out_indices"])
        all_held.extend(split["held_out_indices"])
    assert sorted(all_held) == list(range(2592))


def test_quick_only_technical_no_fake_weather_folds():
    splits = scorer.weather_splits(rows_fixture(True), quick=True)
    assert len(splits) == 1 and splits[0]["technical_only"]
    assert len(splits[0]["train_indices"]) == 36 and splits[0]["held_out_indices"] == []


@pytest.mark.parametrize("mutation", ["fold", "duplicate", "missing", "candidate", "metadata", "index"])
def test_split_rejects_mispaired_metadata(mutation):
    rows = rows_fixture()
    if mutation == "fold": rows[0]["weather_fold"] = 1
    elif mutation == "duplicate": rows[1].update({k: rows[0][k] for k in scorer.pairing.KEY_FIELDS})
    elif mutation == "missing": rows.pop()
    elif mutation == "candidate": rows[0]["candidate_order"].reverse()
    elif mutation == "metadata": rows[0]["metadata_not_model_input"] = False
    else: rows[0]["sample_index"] = 1
    with pytest.raises(ValueError):
        scorer.weather_splits(rows, quick=False)


@pytest.mark.parametrize("extra", ["profile", "weather_seed", "power_delta", "safety_pass", "strehl", "info"])
def test_no_audit_metadata_or_labels_in_input(extra):
    inputs = inputs_fixture(); inputs[extra] = torch.zeros(6)
    with pytest.raises(ValueError, match="只允许"):
        scorer.features(inputs)


def test_absolute_clocks_not_model_features():
    inputs = inputs_fixture(); x, _, _ = scorer.features(inputs)
    changed = deepcopy(inputs); changed["history"][..., 75:78] += 1000
    changed_x, _, _ = scorer.features(changed)
    assert x.shape == (6, 8, 76) and torch.equal(x, changed_x)


def test_train_only_standardization_and_targets_exclude_held_out():
    x, valid, command = scorer.features(inputs_fixture())
    train = torch.tensor([0, 1, 2]); labels = torch.randn(6, 25)
    normalizer = scorer.fit_normalizer(x, valid, command, train)
    changed_x = x.clone(); changed_x[3:] += 1e6
    changed_c = command.clone(); changed_c[3:] -= 1e6
    changed_y = labels.clone(); changed_y[3:] += 1e6
    changed_norm = scorer.fit_normalizer(changed_x, valid, changed_c, train)
    assert all(torch.equal(v, changed_norm[k]) for k, v in normalizer.items())
    before = scorer.training_subset(x, valid, command, labels, train, 10000.)
    after = scorer.training_subset(changed_x, valid, changed_c, changed_y, train, 10000.)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_normalization_padding_and_constant_columns_finite():
    x, valid, command = scorer.features(inputs_fixture())
    normalizer = scorer.fit_normalizer(x, valid, command, torch.arange(3))
    valid[:, :2] = False
    nx, nc = scorer.normalize(x, valid, command, normalizer)
    assert not nx[:, :2].any() and torch.isfinite(nx).all() and torch.isfinite(nc).all()


@pytest.mark.parametrize("kind", ["current", "history"])
def test_model_shape_backward_and_deterministic_forward(kind):
    x, valid, command = scorer.features(inputs_fixture())
    torch.manual_seed(17); model = scorer.CandidateScorer(kind)
    pred = model(x, valid, command)
    assert pred.shape == (6, 23) and torch.equal(pred, model(x, valid, command))
    pred.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_current_model_cannot_use_past_but_history_can():
    x, valid, command = scorer.features(inputs_fixture())
    changed = x.clone(); changed[:, :-1] += 2
    torch.manual_seed(31)
    current = scorer.CandidateScorer("current"); history = scorer.CandidateScorer("history")
    assert torch.equal(current(x, valid, command), current(changed, valid, command))
    assert not torch.equal(history(x, valid, command), history(changed, valid, command))


def test_masked_past_cannot_change_history_model():
    x, valid, command = scorer.features(inputs_fixture()); valid[:, :3] = False
    changed = x.clone(); changed[:, :3] += 10
    model = scorer.CandidateScorer("history")
    assert torch.equal(model(x, valid, command), model(changed, valid, command))


def test_fixed_choice_original_on_negative_or_ties_no_future_filter():
    pred = torch.full((3, 23), -1.)
    pred[1] = 0.; pred[2, 7] = .1
    assert scorer.candidate_choice(pred).tolist() == [0, 0, 9]


@pytest.mark.parametrize("bad", [torch.zeros(2, 25), torch.full((2, 23), float("nan"))])
def test_choice_nonfinite_or_wrong_shape_fails(bad):
    with pytest.raises(ValueError):
        scorer.candidate_choice(bad)


def test_no_overwrite_existing_output(monkeypatch, tmp_path):
    cfg = config(); cfg_path = scorer.pairing.source._project_path(scorer.CONFIG)
    src = scorer.pairing.source
    original_path = src._project_path
    monkeypatch.setattr(scorer, "verify_dataset", lambda quick: tmp_path)
    monkeypatch.setattr(src, "_project_path", lambda p: tmp_path if str(p) == cfg["output_directory"] else original_path(p))
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        scorer.preflight(cfg_path)


def test_cuda_only_preflight_device_contract(monkeypatch, tmp_path):
    # 不依赖正式输出是否已存在，确保本测试能真正抵达 CUDA 检查。
    cfg = config(); src = scorer.pairing.source
    original_path = src._project_path
    monkeypatch.setattr(src, "_project_path", lambda p: tmp_path / "not_created" if str(p) == cfg["output_directory"] else original_path(p))
    monkeypatch.setattr(scorer, "verify_dataset", lambda quick: tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU training was requested"):
        scorer.preflight()
