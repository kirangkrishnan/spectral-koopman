from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # data / training
    trajectories: int = 4000
    training_steps: int = 12        # H
    epochs: int = 3000
    latent_dim: int = 10
    hidden: int = 32
    initial_state_scale: float = 1.0
    spectral: bool = True
    horizons: tuple = (1, 20)       # N in L_spec
    direct_readout_loss: bool = False
    exact_targets: bool = False
    noise: float = 0.01
    seed: int = 8
    # trajectory level
    evaluation_steps: int = 6       # H_e
    error_floor: float = 0.1
    fit: int = 3000
    calibration: int = 1000
    extra_calibration: int = 3000
    audit: int = 500
    final_check: int = 1000
    alpha: float = 0.05
    acceptance: float = 12.0
    tolerance: float = 15.0
    audit_limit: float = 0.01
    domain_fraction: float = 0.25
    gate_min_failures: int = 90
    # pole level
    pole_population: int = 20000
    kappa_quantile: float = 0.99


DEFAULT = Config()

CONFIGS = {
    "vdp": replace(DEFAULT, evaluation_steps=12),
    "arm": replace(DEFAULT, trajectories=5000, epochs=5000, initial_state_scale=2.0),
    "cartpole": replace(DEFAULT, trajectories=20000, epochs=8000, latent_dim=24, hidden=128, initial_state_scale=2.0,
                        horizons=(1, 10)),
    "quadrotor": replace(DEFAULT, trajectories=20000, epochs=5000, latent_dim=24, hidden=128, initial_state_scale=2.0,
                         fit=6000),
    "double_pendulum": replace(DEFAULT, trajectories=20000, epochs=5000, latent_dim=24, hidden=128,
                               initial_state_scale=2.0, fit=6000, horizons=(1, 10)),
}

VARIANTS = {
    "spectral": {},
    "no_loss": {"spectral": False},
    "exact": {"exact_targets": True},
    "direct": {"direct_readout_loss": True},
    "one_step": {"horizons": (1,)},
    "noise05": {"noise": 0.05},
    "no_loss_noise05": {"spectral": False, "noise": 0.05},
    "seed2": {"seed": 2},
    "seed3": {"seed": 3},
}


def get_config(system, variant="spectral"):
    return replace(CONFIGS[system], **VARIANTS[variant])
