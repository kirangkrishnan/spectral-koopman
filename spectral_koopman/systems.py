"""Closed-loop systems used in the experiments. All are sampled at dt = 0.05 s."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import logm

DT = 0.05

# planar quadrotor
QUAD_MASS, QUAD_INERTIA, QUAD_G = 1.0, 0.02, 9.81
QUAD_TORQUE_LIMIT, QUAD_TILT_LIMIT = 2.0, 0.5

# double pendulum on a cart (mujoco); gains are 0.6x to 1.4x of an LQR gain
DP_GEAR = 500.0
DP_LQR = (43.0, 435.0, 2231.0, 75.0, 291.0, 327.0)
DP_XML = Path(__file__).resolve().parent / "assets" / "double_pendulum.xml"

# cart-pole: theta in [-1, 1]^4 is mapped to these gain ranges
CP_LOW = np.array([-5.0, -20.0, -110.0, -35.0])
CP_HIGH = np.array([5.0, 2.0, -20.0, -3.0])


def _stack(values, x):
    return torch.stack(values, dim=-1) if torch.is_tensor(x) else np.stack(values, axis=-1)


@dataclass(frozen=True)
class ClosedLoop:
    name: str
    state_names: tuple
    bounds: tuple           # admissible set, one (low, high) per parameter
    theta_scale: tuple      # network input is theta / theta_scale
    initial_ranges: tuple   # x0 ~ U[-r, r]
    grid_counts: tuple      # points per axis of the evaluation grid
    prediction_weight: float

    @property
    def n(self):
        return len(self.state_names)

    @property
    def p(self):
        return len(self.bounds)

    def feedback(self, x, theta):
        raise NotImplementedError

    def vector_field(self, x, u):
        raise NotImplementedError

    def guard_margin(self, x):
        # > 0 inside the operating set, a run stops when it becomes negative
        return 100.0 - np.max(np.abs(x))

    def sample_x0(self, count, rng, scale=1.0):
        r = scale * np.array(self.initial_ranges)
        return np.asarray([rng.uniform(-r, r) for _ in range(count)])

    def grid(self):
        axes = [np.linspace(-r, r, k) for r, k in zip(self.initial_ranges, self.grid_counts)]
        return np.asarray(list(product(*axes)))

    def rhs(self, x, theta):
        return self.vector_field(x, self.feedback(x, theta))

    def simulate(self, x0, theta, steps, dt=DT):
        """Returns the trajectory (steps + 1, n) and whether it stayed in the operating set."""
        x0, theta = np.asarray(x0, dtype=float), np.asarray(theta, dtype=float)
        traj = np.full((steps + 1, self.n), np.nan)
        traj[0] = x0
        if not np.isfinite(x0).all() or not np.isfinite(theta).all() or self.guard_margin(x0) <= 0:
            return traj, False

        def event(t, x):
            return self.guard_margin(x)
        event.terminal, event.direction = True, -1

        t_eval = np.arange(steps + 1) * dt
        try:
            sol = solve_ivp(lambda t, x: self.rhs(x, theta), (0, t_eval[-1]), x0, t_eval=t_eval,
                            events=event, max_step=dt, method="RK45", rtol=1e-7, atol=1e-9)
        except (ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return traj, False
        if np.size(sol.y):
            traj[:sol.y.shape[1]] = sol.y.T
        ok = (sol.success and not len(sol.t_events[0]) and np.isfinite(traj).all()
              and all(self.guard_margin(x) >= 0 for x in traj))
        return traj, bool(ok)

    def jacobian(self, theta, eps=1e-6):
        # true linearization at the origin, only used for evaluation
        cols = [(self.rhs(eps * e, theta) - self.rhs(-eps * e, theta)) / (2 * eps) for e in np.eye(self.n)]
        return np.stack(cols, axis=1)

    def exact_poles(self, theta):
        return np.linalg.eigvals(self.jacobian(theta))


class VanDerPol(ClosedLoop):
    # u = -(t1 x1 + t2 x2 + t3 x1^3 + t4 x1^2 x2)
    def feedback(self, x, theta):
        u = -(theta[..., 0] * x[..., 0] + theta[..., 1] * x[..., 1]
              + theta[..., 2] * x[..., 0] ** 3 + theta[..., 3] * x[..., 0] ** 2 * x[..., 1])
        return u[..., None]

    def vector_field(self, x, u):
        return np.array([x[1], (1 - x[0] ** 2) * x[1] - x[0] + u[0]])


class TwoLinkArm(ClosedLoop):
    # decentralized PD, torques saturated at 30
    def feedback(self, x, theta):
        u = _stack((-(theta[..., 0] * x[..., 0] + theta[..., 1] * x[..., 2]),
                    -(theta[..., 2] * x[..., 1] + theta[..., 3] * x[..., 3])), x)
        return u.clip(-30, 30)

    def vector_field(self, x, u):
        d = x[0] - x[1]
        mass = np.array([[2.0, np.cos(d)], [np.cos(d), 1.0]])
        bias = np.array([np.sin(d) * x[3] ** 2 - 19.62 * np.sin(x[0]),
                         -np.sin(d) * x[2] ** 2 - 9.81 * np.sin(x[1])])
        return np.r_[x[2:], np.linalg.solve(mass, u - bias)]

    def guard_margin(self, x):
        return 3.0 - np.max(np.abs(x))


class CartPole(ClosedLoop):
    # u = -(k1 x + k2 dx + k3 sin(phi) + k4 dphi), force saturated at 20
    @staticmethod
    def gains(theta):
        if torch.is_tensor(theta):
            low, high = theta.new_tensor(CP_LOW), theta.new_tensor(CP_HIGH)
        else:
            low, high, theta = CP_LOW, CP_HIGH, np.asarray(theta)
        return low + 0.5 * (theta + 1) * (high - low)

    def feedback(self, x, theta):
        k = self.gains(theta)
        sin = torch.sin if torch.is_tensor(x) else np.sin
        u = -(k[..., 0] * x[..., 0] + k[..., 1] * x[..., 1] + k[..., 2] * sin(x[..., 2]) + k[..., 3] * x[..., 3])
        return u.clip(-20, 20)[..., None]

    def vector_field(self, x, u):
        v, phi, w = x[1], x[2], x[3]
        s, c = np.sin(phi), np.cos(phi)
        temp = (u[0] + 0.1 * 0.5 * w ** 2 * s) / 1.1
        phi_acc = (9.81 * s - c * temp) / (0.5 * (4 / 3 - 0.1 * c ** 2 / 1.1))
        x_acc = temp - 0.1 * 0.5 * phi_acc * c / 1.1
        return np.array([v, x_acc, w, phi_acc])

    def guard_margin(self, x):
        return min(10 - abs(x[0]), np.deg2rad(20) - abs(x[2]), 100 - np.max(np.abs(x)))


class PlanarQuadrotor(ClosedLoop):
    # state (px, pz, phi, vx, vz, omega); cascaded PD with theta = (kpx, kdx, kpz, kdz, kphi, komega)
    def feedback(self, x, theta):
        kpx, kdx, kpz, kdz, kphi, kw = (theta[..., i] for i in range(6))
        thrust = (-QUAD_MASS * (kpz * x[..., 1] + kdz * x[..., 4])).clip(-QUAD_MASS * QUAD_G, QUAD_MASS * QUAD_G)
        tilt = ((kpx * x[..., 0] + kdx * x[..., 3]) / QUAD_G).clip(-QUAD_TILT_LIMIT, QUAD_TILT_LIMIT)
        torque = (QUAD_INERTIA * (-kphi * (x[..., 2] - tilt) - kw * x[..., 5])).clip(-QUAD_TORQUE_LIMIT,
                                                                                      QUAD_TORQUE_LIMIT)
        return _stack((thrust, torque), x)

    def vector_field(self, x, u):
        f = QUAD_MASS * QUAD_G + u[0]
        return np.array([x[3], x[4], x[5], -f / QUAD_MASS * np.sin(x[2]),
                         f / QUAD_MASS * np.cos(x[2]) - QUAD_G, u[1] / QUAD_INERTIA])

    def guard_margin(self, x):
        return min(10 - abs(x[0]), 10 - abs(x[1]), 1.2 - abs(x[2]), 100 - np.max(np.abs(x)))


_MJ = {}


def mujoco_handles():
    if not _MJ:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(DP_XML))
        _MJ.update(lib=mujoco, model=model, data=mujoco.MjData(model))
    return _MJ["lib"], _MJ["model"], _MJ["data"]


class DoublePendulumOnCart(ClosedLoop):
    # state (x, phi1, phi2, dx, dphi1, dphi2), u = -theta . x (saturated at 500 N), control updated every 0.01 s.
    # No equations, so the "exact" linearization is a finite difference of the sampled map.
    def feedback(self, x, theta):
        u = -(theta * x).sum(-1)
        return u.clip(-DP_GEAR, DP_GEAR)[..., None]

    def guard_margin(self, x):
        return min(1.5 - abs(x[0]), 0.5 - abs(x[1]), 0.5 - abs(x[2]), 100 - np.max(np.abs(x)))

    def _advance(self, x, theta, steps, substeps):
        mujoco, model, data = mujoco_handles()
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = x[:3], x[3:]
        out = np.full((steps + 1, 6), np.nan)
        out[0] = x
        for k in range(steps):
            for _ in range(substeps):
                data.ctrl[0] = self.feedback(np.r_[data.qpos, data.qvel], theta)[0] / DP_GEAR
                mujoco.mj_step(model, data)
            out[k + 1] = np.r_[data.qpos, data.qvel]
            if not np.isfinite(out[k + 1]).all() or self.guard_margin(out[k + 1]) < 0:
                out[k + 1:] = np.nan
                return out, False
        return out, True

    @staticmethod
    def _substeps(dt):
        _, model, _ = mujoco_handles()
        k = int(round(dt / model.opt.timestep))
        if k < 1 or abs(k * model.opt.timestep - dt) > 1e-9:
            raise ValueError("dt must be a multiple of the mujoco time step (0.01 s)")
        return k

    def simulate(self, x0, theta, steps, dt=DT):
        k = self._substeps(dt)
        x0, theta = np.asarray(x0, dtype=float), np.asarray(theta, dtype=float)
        if not np.isfinite(x0).all() or not np.isfinite(theta).all() or self.guard_margin(x0) <= 0:
            traj = np.full((steps + 1, self.n), np.nan)
            traj[0] = x0
            return traj, False
        return self._advance(x0, theta, steps, k)

    def map_jacobian(self, theta, dt=DT, eps=1e-6):
        k = self._substeps(dt)
        cols = [(self._advance(eps * e, theta, 1, k)[0][1] - self._advance(-eps * e, theta, 1, k)[0][1]) / (2 * eps)
                for e in np.eye(self.n)]
        return np.stack(cols, axis=1)

    def jacobian(self, theta, eps=1e-6, dt=DT):
        return np.real(logm(self.map_jacobian(theta, dt, eps))) / dt


_POLY_SET = ((0.0, 5.0), (0.0, 5.0), (-1.0, 1.0), (-1.0, 1.0))

SYSTEMS = {
    "vdp": VanDerPol("vdp", ("x1", "x2"), _POLY_SET, (1.0,) * 4, (1.0, 1.0), (5, 5), 2.0),
    "arm": TwoLinkArm("arm", ("q1", "q2", "dq1", "dq2"), ((0.0, 40.0), (0.0, 20.0), (0.0, 20.0), (0.0, 20.0)),
                      (40.0, 20.0, 20.0, 20.0), (0.35, 0.35, 1.0, 1.0), (3, 3, 2, 2), 4.0),
    "cartpole": CartPole("cartpole", ("x", "dx", "phi", "dphi"), ((-1.0, 1.0),) * 4, (1.0,) * 4,
                         (0.5, 0.5, 0.3, 0.8), (3,) * 4, 8.0),
    "quadrotor": PlanarQuadrotor("quadrotor", ("px", "pz", "phi", "vx", "vz", "omega"),
                                 ((0.5, 6.0), (0.5, 5.0), (0.5, 6.0), (0.5, 5.0), (5.0, 120.0), (1.0, 25.0)),
                                 (6.0, 5.0, 6.0, 5.0, 120.0, 25.0), (0.5, 0.5, 0.3, 0.5, 0.5, 1.0), (2,) * 6, 4.0),
    "double_pendulum": DoublePendulumOnCart("double_pendulum", ("x", "phi1", "phi2", "dx", "dphi1", "dphi2"),
                                            tuple((0.6 * k, 1.4 * k) for k in DP_LQR),
                                            tuple(1.4 * k for k in DP_LQR),
                                            (0.2, 0.05, 0.05, 0.2, 0.2, 0.2), (2,) * 6, 4.0),
}


def get_system(name):
    return SYSTEMS[name]
