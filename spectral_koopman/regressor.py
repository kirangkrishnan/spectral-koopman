"""Baseline: direct regression of the linearization, J(theta) = I + G(theta / theta_scale).

Trained by least squares on the same identified targets and training values as the spectral model, in coordinates
scaled by the initial-state ranges. Keeps the epoch with the lowest loss on 10% held-out values.
"""
import numpy as np
import torch
from torch import nn


class DirectRegressor(nn.Module):
    def __init__(self, system, hidden=128, layers=3):
        super().__init__()
        self.n = system.n
        widths = [system.p] + [hidden] * layers
        body = []
        for a, b in zip(widths[:-1], widths[1:]):
            body += [nn.Linear(a, b), nn.Tanh()]
        self.net = nn.Sequential(*body, nn.Linear(hidden, self.n * self.n))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("theta_scale", torch.tensor(system.theta_scale, dtype=torch.float32))

    def readout(self, theta):
        g = self.net(theta / self.theta_scale.to(theta.dtype)).reshape(-1, self.n, self.n)
        return torch.eye(self.n, dtype=theta.dtype) + g

    def predict(self, x0, theta, steps):
        # linear rollout x_{k+1} = J x_k, the only trajectory prediction it has
        J, x, out = self.readout(theta), x0, []
        for _ in range(steps):
            x = torch.bmm(J, x.unsqueeze(-1)).squeeze(-1)
            out.append(x)
        return torch.stack(out, dim=1)


def train_regressor(system, thetas, targets, mask, seed=8, epochs=2000, hidden=128, layers=3, batch_size=256,
                    val_frac=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(idx))))
    i_val, i_fit = idx[:n_val], idx[n_val:]
    r = torch.tensor(np.asarray(system.initial_ranges, dtype=float))
    scale = lambda m: m / r[:, None] * r[None, :]
    th = torch.tensor(thetas, dtype=torch.float64)
    tg = scale(torch.tensor(targets, dtype=torch.float64))
    model = DirectRegressor(system, hidden, layers).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100)
    loss_of = lambda i: (scale(model.readout(th[i])) - tg[i]).square().sum((1, 2)).mean()
    best, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        order = i_fit[rng.permutation(len(i_fit))]
        losses = []
        for start in range(0, len(order), batch_size):
            i = torch.as_tensor(order[start:start + batch_size])
            opt.zero_grad()
            loss = loss_of(i)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            val = float(loss_of(torch.as_tensor(i_val)))
        sched.step(val)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val": val})
        if val < best:
            best, best_state = val, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), history


def save_regressor(model, path, system, hidden=128, layers=3):
    torch.save({"kind": "regressor", "system": system.name, "hidden": hidden, "layers": layers,
                "state_dict": model.state_dict()}, path)
