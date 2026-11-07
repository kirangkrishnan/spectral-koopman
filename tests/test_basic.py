from pathlib import Path

import numpy as np
import pytest
import torch

from spectral_koopman import SYSTEMS
from spectral_koopman.conformal import conformal_quantile
from spectral_koopman.identification import Budget, identify
from spectral_koopman.model import KoopmanModel, load_model, poles
from spectral_koopman.pole_level import pole_bounds
from spectral_koopman.trajectory_level import load_region

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_quantile():
    s = np.arange(1.0, 20.0)
    assert conformal_quantile(s, 0.05) == 19.0
    assert conformal_quantile(s[:10], 0.05) == np.inf


def test_coverage():
    rng = np.random.default_rng(0)
    hits = [rng.normal() <= conformal_quantile(rng.normal(size=99), 0.1) for _ in range(4000)]
    assert 0.88 < np.mean(hits) < 0.93


def test_identify_vdp():
    vdp = SYSTEMS["vdp"]
    theta = np.array([2.0, 3.0, 0.0, 0.0])
    f = identify(vdp, theta, Budget(16, 20, amplitude=0.01, noise=0.0, purpose="test"))
    lam = np.log(np.linalg.eigvals(f).astype(complex)) / 0.05
    assert np.abs(np.sort_complex(lam) - np.sort_complex(vdp.exact_poles(theta))).max() < 0.02


def test_readout_exact():
    # if K V = V F then V^+ K V = F
    model = KoopmanModel(SYSTEMS["vdp"], latent_dim=6, hidden=8).double()
    th = torch.tensor([[1.0, 2.0, 0.5, -0.5]], dtype=torch.float64)
    v = model.encoder_jacobian(th)[0]
    f = torch.tensor([[0.9, 0.1], [-0.2, 0.8]], dtype=torch.float64)
    k = v @ f @ torch.linalg.pinv(v)
    assert torch.allclose(torch.linalg.solve(v.T @ v, v.T @ k @ v), f, atol=1e-10)


@pytest.mark.parametrize("name", ["vdp", "arm", "cartpole", "quadrotor"])
def test_released(name):
    d = RESULTS / name / "spectral"
    region = load_region(d / "region.npz")
    th = region.sample(50, np.random.default_rng(0))
    assert np.all(region.upper(th) <= 12.0)
    assert poles(load_model(d / "model.pth", SYSTEMS[name]), th, 0.05).shape == (50, SYSTEMS[name].n)


def test_pole_bounds_vdp():
    labels = dict(np.load(RESULTS / "vdp" / "spectral" / "pole_labels.npz"))
    _, res = pole_bounds(labels, SYSTEMS["vdp"].bounds)
    assert res["rate"]["audit_passed"] and res["damping"]["audit_passed"]
    assert abs(res["rate"]["coverage"] - 0.969429) < 1e-6
