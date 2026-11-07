"""NMPC baseline for the one-shot experiment.

Same stage cost as the selection, horizon 20 steps, terminal weight 10, saturated inputs and the operating set as
state constraints. CasADi/IPOPT single shooting with RK4 where we have equations; for the mujoco double pendulum,
single shooting through the simulator with SLSQP.
"""
import time

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .selection import BETA
from .systems import DP_GEAR, DT, QUAD_G, QUAD_INERTIA, QUAD_MASS, QUAD_TORQUE_LIMIT, mujoco_handles

H_MPC = 20
U_LIMITS = {"vdp": ([-10.0], [10.0]), "arm": ([-30.0] * 2, [30.0] * 2), "cartpole": ([-20.0], [20.0]),
            "quadrotor": ([-QUAD_MASS * QUAD_G, -QUAD_TORQUE_LIMIT], [QUAD_MASS * QUAD_G, QUAD_TORQUE_LIMIT]),
            "double_pendulum": ([-DP_GEAR], [DP_GEAR])}
X_LIMITS = {"vdp": [99.0] * 2, "arm": [2.95] * 4, "cartpole": [9.9, 99.0, 0.34, 99.0],
            "quadrotor": [9.9, 9.9, 1.18, 99.0, 99.0, 99.0]}   # operating set, slightly tightened


def open_loop(lib, name, x, u):
    # plant under input u, written so that it works with numpy and casadi
    sin, cos = lib.sin, lib.cos
    if name == "vdp":
        return [x[1], (1 - x[0] ** 2) * x[1] - x[0] + u[0]]
    if name == "arm":
        d = x[0] - x[1]
        b1 = sin(d) * x[3] ** 2 - 19.62 * sin(x[0])
        b2 = -sin(d) * x[2] ** 2 - 9.81 * sin(x[1])
        r1, r2, det = u[0] - b1, u[1] - b2, 2 - cos(d) ** 2
        return [x[2], x[3], (r1 - cos(d) * r2) / det, (2 * r2 - cos(d) * r1) / det]
    if name == "quadrotor":
        f = QUAD_MASS * QUAD_G + u[0]
        return [x[3], x[4], x[5], -f / QUAD_MASS * sin(x[2]), f / QUAD_MASS * cos(x[2]) - QUAD_G, u[1] / QUAD_INERTIA]
    s, c = sin(x[2]), cos(x[2])   # cartpole
    temp = (u[0] + 0.1 * 0.5 * x[3] ** 2 * s) / 1.1
    acc = (9.81 * s - c * temp) / (0.5 * (4 / 3 - 0.1 * c ** 2 / 1.1))
    return [x[1], temp - 0.1 * 0.5 * acc * c / 1.1, x[3], acc]


def plant_step(system, x, u, dt=DT):
    # one sample of the true plant with u held; None if it leaves the operating set
    if system.name == "double_pendulum":
        mujoco, model, data = mujoco_handles()
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = x[:3], x[3:]
        data.ctrl[0] = float(u[0]) / DP_GEAR
        for _ in range(int(round(dt / model.opt.timestep))):
            mujoco.mj_step(model, data)
        nxt = np.r_[data.qpos, data.qvel]
    else:
        nxt = solve_ivp(lambda t, s: np.array(open_loop(np, system.name, s, u)), (0, dt), x,
                        rtol=1e-7, atol=1e-9, max_step=dt).y[:, -1]
    return nxt if np.isfinite(nxt).all() and system.guard_margin(nxt) >= 0 else None


class CasadiNMPC:
    def __init__(self, system, dt=DT, horizon=H_MPC, terminal=10.0, max_iter=60):
        n, (lo, hi) = system.n, U_LIMITS[system.name]
        self.nu, self.H = len(lo), horizon
        self.lo, self.hi = np.array(lo), np.array(hi)
        f = lambda x, u: ca.vertcat(*open_loop(ca, system.name, x, u))

        def rk4(x, u):
            k1 = f(x, u)
            k2 = f(x + 0.5 * dt * k1, u)
            k3 = f(x + 0.5 * dt * k2, u)
            k4 = f(x + dt * k3, u)
            return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        U, x0 = ca.SX.sym("u", horizon * self.nu), ca.SX.sym("x0", n)
        x, J, xs = x0, 0, []
        Q, R = ca.DM(np.eye(n)), ca.DM(BETA[system.name] * np.eye(self.nu))
        for k in range(horizon):
            uk = U[k * self.nu:(k + 1) * self.nu]
            x = rk4(x, uk)
            xs.append(x)
            J += ca.mtimes([x.T, Q, x]) + ca.mtimes([uk.T, R, uk])
        J += terminal * ca.mtimes([x.T, Q, x])
        xb = np.tile(X_LIMITS[system.name], horizon)
        self.g_bounds = {"lbg": -xb, "ubg": xb}
        opts = {"print_time": False, "error_on_fail": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                "ipopt.max_iter": max_iter, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-3,
                "ipopt.acceptable_iter": 3, "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-3,
                "ipopt.bound_push": 1e-8, "ipopt.bound_frac": 1e-8}
        self.solver = ca.nlpsol("nmpc", "ipopt", {"x": U, "p": x0, "f": J, "g": ca.vertcat(*xs)}, opts)
        self.warm = np.zeros((horizon, self.nu))
        self.lam = None

    def solve(self, x):
        prev = self.warm.copy()
        kw = {"x0": prev.reshape(-1), "p": np.asarray(x, float), "lbx": np.tile(self.lo, self.H),
              "ubx": np.tile(self.hi, self.H), **self.g_bounds}
        if self.lam is not None:
            kw["lam_x0"] = self.lam
        try:
            sol = self.solver(**kw)
            cand = np.asarray(sol["x"], dtype=float).reshape(self.H, self.nu)
            ok = bool(self.solver.stats().get("success", False)) and np.isfinite(cand).all()
        except Exception:
            ok = False
        if ok:
            plan, self.lam = cand, np.asarray(sol["lam_x"], dtype=float).reshape(-1)
        else:
            plan, self.lam = prev, None
        self.warm[:-1], self.warm[-1] = plan[1:], 0.0
        return np.clip(plan[0], self.lo, self.hi)


class MujocoNMPC:
    # warm start: shifted previous plan or the inputs of a fixed stabilizing theta, whichever is cheaper
    def __init__(self, system, warm_theta, dt=DT, horizon=H_MPC):
        self.system, self.dt, self.H = system, dt, horizon
        self.beta, self.warm_theta, self.warm = BETA[system.name], np.asarray(warm_theta), None

    def cost(self, x, us):
        J = 0.0
        for k, u in enumerate(us):
            x = plant_step(self.system, x, [u], self.dt)
            if x is None:
                return 1e6 * (len(us) - k)
            J += x @ x + self.beta * u ** 2
        return J + 10 * x @ x

    def feedback_inputs(self, x):
        us = []
        for _ in range(self.H):
            u = float(self.system.feedback(x, self.warm_theta)[0])
            us.append(u)
            x = plant_step(self.system, x, [u], self.dt)
            if x is None:
                return np.array(us + [0.0] * (self.H - len(us)))
        return np.array(us)

    def solve(self, x):
        guesses = [self.feedback_inputs(x)] + ([] if self.warm is None else [self.warm])
        u0 = min(guesses, key=lambda g: self.cost(x, g))
        res = minimize(lambda g: self.cost(x, g), u0, method="SLSQP", bounds=[(-DP_GEAR, DP_GEAR)] * self.H,
                       options={"maxiter": 30, "ftol": 1e-6})
        best = min([u0, res.x], key=lambda g: self.cost(x, g))
        self.warm = np.r_[best[1:], best[-1]]
        return np.array([best[0]])


def run_nmpc(system, x0, warm_theta=None, duration=20.0, dt=DT):
    ctrl = MujocoNMPC(system, warm_theta, dt) if system.name == "double_pendulum" else CasadiNMPC(system, dt)
    steps = int(round(duration / dt))
    nu = len(U_LIMITS[system.name][0])
    traj, us = np.full((steps + 1, system.n), np.nan), np.full((steps + 1, nu), np.nan)
    traj[0], x, times = x0, np.asarray(x0, float), []
    for k in range(steps):
        t0 = time.perf_counter()
        u = ctrl.solve(x)
        times.append(time.perf_counter() - t0)
        us[k] = u
        x = plant_step(system, x, np.asarray(u, float), dt)
        if x is None:
            break
        traj[k + 1] = x
    return traj, us, x is not None, times
