"""Training with the prediction loss and (optionally) the spectral-consistency loss.

L_spec = sum_N ||K^N V - V F^N||^2 / ||V||^2 + ||W V - I||^2, with F the identified linearization.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import KoopmanModel

mse = nn.functional.mse_loss


def prediction_loss(model, x, theta, w_pred):
    # x: (B, H + 1, n). Reconstruction, forward rollout with K, backward rollout with B from x_H,
    # K B ~ I and a small penalty on both operators.
    H = x.shape[1] - 1
    z = model.encode(x[:, 0], theta)
    k, b = model.operators(theta)
    loss = mse(model.decode(z, theta), x[:, 0])
    eye = torch.eye(model.latent_dim, device=k.device).expand_as(k)
    loss = loss + 0.1 * mse(k @ b, eye)
    loss = loss + 1e-4 * (k.square().sum((1, 2)).mean() + b.square().sum((1, 2)).mean())
    for op, zk, steps in ((k, z, range(1, H + 1)),
                          (b, model.encode(x[:, -1], theta), range(H - 1, -1, -1))):
        for i in steps:
            zk = torch.bmm(op, zk.unsqueeze(-1)).squeeze(-1)
            loss = loss + (w_pred * mse(model.decode(zk, theta), x[:, i])
                           + mse(zk, model.encode(x[:, i], theta))) / H
    return loss, k


def spectral_loss(model, k, theta, target, horizons=(1, 20), mask=None, direct=False):
    v = model.encoder_jacobian(theta)
    norm = v.square().sum((1, 2)) + 1e-6
    res = 0
    if direct:
        # ablation: fit M = V^+ K V to the target instead of the intertwining residuals
        vt = v.transpose(-1, -2)
        res = (torch.linalg.solve(vt @ v, vt @ k @ v) - target).square().sum((1, 2))
        horizons = ()
    for N in horizons:
        r = torch.linalg.matrix_power(k, N) @ v - v @ torch.linalg.matrix_power(target, N)
        res = res + r.square().sum((1, 2)) / norm
    eye = torch.eye(v.shape[-1], device=v.device).expand(v.shape[0], -1, -1)
    consistency = (model.decoder_jacobian(theta) @ v - eye).square().sum((1, 2))
    if mask is None:
        return res.mean() + consistency.mean()
    return (res * mask).sum() / mask.sum().clamp(min=1.0) + consistency.mean()


def train(system, trajs, thetas, epochs, seed, latent_dim=10, hidden=32, targets=None, mask=None,
          horizons=(1, 20), direct=False, batch_size=128, device="cpu", log_every=50):
    """targets=None trains without the spectral loss."""
    torch.manual_seed(seed)
    spectral = targets is not None
    small_init = spectral or system.name in ("arm", "cartpole")
    model = KoopmanModel(system, latent_dim, hidden, small_operator_init=small_init).to(device)
    if targets is None:
        targets = np.zeros((len(thetas), system.n, system.n))
    w = np.ones(len(thetas)) if mask is None else np.asarray(mask, dtype=float)
    data = TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in (trajs, thetas, targets, w)))
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100)
    history = []
    for epoch in range(epochs):
        losses = []
        for x, th, tg, valid in loader:
            x, th, tg, valid = (t.to(device) for t in (x, th, tg, valid))
            opt.zero_grad()
            loss, k = prediction_loss(model, x, th, system.prediction_weight)
            if spectral:
                loss = loss + spectral_loss(model, k, th, tg, horizons, None if mask is None else valid, direct)
            if not torch.isfinite(loss):
                raise RuntimeError("loss is not finite")
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        avg = float(np.mean(losses))
        sched.step(avg)
        history.append({"epoch": epoch + 1, "loss": avg, "lr": opt.param_groups[0]["lr"]})
        if log_every and (epoch == 0 or (epoch + 1) % log_every == 0):
            print(f"{system.name}: epoch {epoch + 1}/{epochs}, loss {avg:.6g}", flush=True)
    return model.cpu().eval(), history
