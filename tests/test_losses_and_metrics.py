import torch

from src.evaluate_compensation import compute_batch_metrics
from src.losses import compensation_loss, compensate_field, complex_field_from_intensity_phase


def test_compensate_field_recovers_clean_phase_when_correction_matches_turbulence():
    intensity = torch.ones(1, 1, 4, 4)
    phase_turb = torch.full((1, 1, 4, 4), 0.75)
    phase_corr = torch.full((1, 1, 4, 4), 0.75)

    real, imag = compensate_field(intensity, phase_turb, phase_corr)

    assert torch.allclose(real, torch.ones_like(real), atol=1e-6)
    assert torch.allclose(imag, torch.zeros_like(imag), atol=1e-6)


def test_compensation_loss_is_lower_for_correct_phase_correction():
    intensity = torch.ones(1, 1, 4, 4)
    phase_turb = torch.full((1, 1, 4, 4), 0.5)
    target_intensity = torch.ones(1, 1, 4, 4)
    target_phase = torch.zeros(1, 1, 4, 4)

    good = compensation_loss(
        phi_corr=torch.full((1, 1, 4, 4), 0.5),
        input_intensity=intensity,
        input_phase=phase_turb,
        target_intensity=target_intensity,
        target_phase=target_phase,
    )
    bad = compensation_loss(
        phi_corr=torch.zeros(1, 1, 4, 4),
        input_intensity=intensity,
        input_phase=phase_turb,
        target_intensity=target_intensity,
        target_phase=target_phase,
    )

    assert good["total"].item() < bad["total"].item()


def test_compute_batch_metrics_reports_perfect_recovery():
    intensity = torch.ones(1, 1, 8, 8)
    phase = torch.zeros(1, 1, 8, 8)
    real, imag = complex_field_from_intensity_phase(intensity, phase)

    metrics = compute_batch_metrics(real, imag, real, imag)

    assert metrics["complex_corr"] == 1.0
    assert metrics["intensity_mse"] == 0.0
    assert metrics["phase_rmse"] == 0.0
    assert metrics["strehl_ratio"] == 1.0
    assert metrics["rc2"] == 0.0
