"""Feedback-parameterized Koopman model.

x_hat_k = psi(K^k phi(x0)), with encoder phi, operator K and decoder psi all conditioned on theta / theta_scale.
The read-out M = V^+ K V (V = d phi / dx at 0) gives the model's linearization and its poles log(eig(M)) / dt.
"""
import numpy as np
import torch
from torch import nn


class KoopmanModel(nn.Module):
    def __init__(self, system, latent_dim=10, hidden=32, small_operator_init=False):
        super().__init__()
        n, p = system.n, system.p
        self.latent_dim = latent_dim
        # NB: the order in which the modules are created fixes the initialization
        self.encoder = nn.Sequential(nn.Linear(n + p, hidden), nn.Tanh(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim + p, hidden), nn.Tanh(), nn.Linear(hidden, n))
        self.operator_net = nn.Sequential(nn.Linear(p, hidden), nn.ReLU(), nn.Linear(hidden, 2 * latent_dim ** 2))
        self.register_buffer("theta_scale", torch.tensor(system.theta_scale))
        if small_operator_init:
            # start with K ~ 0, otherwise the N-step terms of the spectral loss blow up
            nn.init.uniform_(self.operator_net[-1].weight, -0.01, 0.01)
            nn.init.uniform_(self.operator_net[-1].bias, -0.01, 0.01)

    def encode(self, x, theta):
        s = theta / self.theta_scale
        return self.encoder(torch.cat((x, s), -1)) - self.encoder(torch.cat((torch.zeros_like(x), s), -1))

    def decode(self, z, theta):
        s = theta / self.theta_scale
        return self.decoder(torch.cat((z, s), -1)) - self.decoder(torch.cat((torch.zeros_like(z), s), -1))

    def operators(self, theta):
        # forward operator K and backward operator B (B is only used in training)
        m = self.latent_dim
        out = self.operator_net(theta / self.theta_scale)
        return out[..., :m * m].reshape(-1, m, m), out[..., m * m:].reshape(-1, m, m)

    def predict(self, x0, theta, steps, latent=False):
        z = self.encode(x0, theta)
        k, _ = self.operators(theta)
        out = []
        for _ in range(steps):
            z = torch.bmm(k, z.unsqueeze(-1)).squeeze(-1)
            out.append(z if latent else self.decode(z, theta))
        return torch.stack(out, dim=1)

    def encoder_jacobian(self, theta):
        return _jac_at_zero(self.encoder, theta / self.theta_scale)    # V, (B, m, n)

    def decoder_jacobian(self, theta):
        return _jac_at_zero(self.decoder, theta / self.theta_scale)    # W, (B, n, m)

    def readout(self, theta):
        k, _ = self.operators(theta)
        v = self.encoder_jacobian(theta)
        vt = v.transpose(-1, -2)
        return torch.linalg.solve(vt @ v, vt @ k @ v)                    # M = V^+ K V


def _jac_at_zero(net, s):
    # Jacobian of x -> net([x, s]) at x = 0, net = Linear-Tanh-Linear
    first, second = net[0], net[2]
    n = first.in_features - s.shape[-1]
    h = s @ first.weight[:, n:].T + first.bias
    return torch.einsum("oh,bh,hi->boi", second.weight, 1 - torch.tanh(h).square(), first.weight[:, :n])


def _t(thetas):
    return torch.as_tensor(np.atleast_2d(thetas), dtype=torch.float64)


def readout_matrices(model, thetas):
    with torch.no_grad():
        return model.readout(_t(thetas)).numpy()


def poles(model, thetas, dt):
    """Continuous-time poles of the read-out (model in float64)."""
    with torch.no_grad():
        return (torch.log(torch.linalg.eigvals(model.readout(_t(thetas)))) / dt).numpy()


def slowest_rate(lam):
    return np.atleast_2d(lam).real.max(1)


def smallest_damping(lam, stable_only=False):
    # min_i -Re(l_i)/|l_i|; with stable_only only poles with Re < 0 count (1 if there are none)
    lam = np.atleast_2d(lam)
    ratio = -lam.real / np.maximum(np.abs(lam), 1e-12)
    if not stable_only:
        return np.clip(ratio.min(1), -1, 1)
    ratio = np.where(lam.real < 0, ratio, np.inf).min(1)
    return np.clip(np.where(np.isfinite(ratio), ratio, 1.0), -1, 1)


def rate_damping(model, thetas, dt):
    lam = poles(model, thetas, dt)
    return slowest_rate(lam), smallest_damping(lam)


def kappa(model, thetas):
    # condition number of the eigenvector matrix of M, large when learned poles nearly merge
    return np.array([np.linalg.cond(np.linalg.eig(m)[1]) for m in readout_matrices(model, thetas)])


def save_model(model, path, system, **extra):
    torch.save({"kind": "koopman", "system": system.name, "latent_dim": model.latent_dim,
                "hidden": model.encoder[0].out_features, "state_dict": model.state_dict(), **extra}, path)


def load_model(path, system):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["system"] != system.name:
        raise ValueError(f"{path} is a {ckpt['system']} model, not {system.name}")
    if ckpt["kind"] == "regressor":
        from .regressor import DirectRegressor
        model = DirectRegressor(system, ckpt["hidden"], ckpt["layers"])
    else:
        model = KoopmanModel(system, ckpt["latent_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model.double().eval()
