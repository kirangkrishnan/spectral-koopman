"""Conformal bounds U(theta) = exp(g(theta) + q) - c.

g is a ridge regression on random Fourier features that predicts log(s + c) for a score s >= 0, and q is the split
conformal quantile of log(s + c) - g(theta) on separate calibration values. Trajectory level: s = e(theta) in %,
c = 1, censored at 1000%. Pole level: s = |measured - learned| rate or damping, c = 1e-3.
"""
import numpy as np
from scipy.stats import beta
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, Ridge


def conformal_quantile(scores, alpha):
    # ceil((n + 1)(1 - alpha))-th smallest score, inf if that is more than n
    scores = np.asarray(scores, dtype=float)
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else float(np.partition(scores, k - 1)[k - 1])


def clopper_pearson_upper(fails, n, conf=0.95):
    return 1.0 if fails == n else float(beta.ppf(conf, fails + 1, n - fails))


def clopper_pearson_lower(fails, n, conf=0.95):
    return 0.0 if fails == 0 else float(beta.ppf(1 - conf, fails, n - fails + 1))


class RFFModel:
    """Linear model on random Fourier features (RBF, gamma = 1) of theta scaled to [0, 1]^p.

    Fitted with sklearn, but only the arrays are kept so that it can be saved without pickling.
    """

    def __init__(self, bounds, n_features=256, seed=8):
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_features, self.seed = n_features, seed
        self.W = self.b = self.coef = self.intercept = None

    def scaled(self, thetas):
        return (np.asarray(thetas, dtype=float) - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel()

    def features(self, thetas):
        proj = self.scaled(thetas) @ self.W + self.b
        return np.cos(proj) * (np.sqrt(2.0) / np.sqrt(self.n_features))   # same as sklearn's RBFSampler

    def __call__(self, thetas):
        return self.features(thetas) @ self.coef + self.intercept

    def _fit(self, est, thetas, y):
        rff = RBFSampler(gamma=1.0, n_components=self.n_features, random_state=self.seed)
        est.fit(rff.fit_transform(self.scaled(thetas)), y)
        self.W, self.b = rff.random_weights_, rff.random_offset_
        self.coef, self.intercept = np.ravel(est.coef_), float(np.ravel(est.intercept_)[0])
        return self

    def fit_ridge(self, thetas, y):
        return self._fit(Ridge(alpha=1e-3), thetas, y)

    def fit_logistic(self, thetas, labels):
        return self._fit(LogisticRegression(C=1.0, max_iter=5000), thetas, labels)

    def arrays(self, prefix):
        return {f"{prefix}_W": self.W, f"{prefix}_b": self.b, f"{prefix}_coef": self.coef,
                f"{prefix}_intercept": np.asarray(self.intercept)}

    @classmethod
    def from_arrays(cls, bounds, a, prefix):
        m = cls(bounds, len(a[f"{prefix}_b"]))
        m.W, m.b = a[f"{prefix}_W"], a[f"{prefix}_b"]
        m.coef, m.intercept = a[f"{prefix}_coef"], float(a[f"{prefix}_intercept"])
        return m


class ConformalBound:
    def __init__(self, bounds, offset, fit_cap, cal_cap=None, seed=8):
        self.g = RFFModel(bounds, seed=seed)
        self.c, self.fit_cap, self.cal_cap = offset, fit_cap, cal_cap
        self.q = None

    def _log(self, s):
        return np.log(np.asarray(s, dtype=float) + self.c)

    def fit(self, thetas, s):
        self.g.fit_ridge(thetas, self._log(np.minimum(s, self.fit_cap)))   # failed runs have s = inf
        return self

    def residuals(self, thetas, s):
        s = np.asarray(s, dtype=float)
        if self.cal_cap is not None:
            s = np.minimum(s, self.cal_cap)
        with np.errstate(invalid="ignore"):
            return self._log(s) - self.g(thetas)

    def calibrate(self, thetas, s, alpha):
        self.q = conformal_quantile(self.residuals(thetas, s), alpha)
        return self

    def upper(self, thetas):
        log_u = self.g(thetas) + self.q
        with np.errstate(over="ignore", invalid="ignore"):
            u = np.maximum(0.0, np.exp(log_u) - self.c)
        if self.cal_cap is not None:
            # censored scores: the bound says nothing above the cap
            u = np.where(log_u >= self._log(self.cal_cap), np.inf, u)
        x = self.g.scaled(thetas)
        inside = np.all((x >= 0) & (x <= 1), axis=1)
        return np.where(inside & np.isfinite(u), u, np.inf)


class FailureGate:
    """Kernel logistic classifier of failed runs, used only when choosing the domain D.

    The threshold is calibrated on the failed calibration values (class-conditional conformal), so that a failing
    value passes with probability at most alpha.
    """

    def __init__(self, bounds, alpha=0.1):
        self.clf = RFFModel(bounds)
        self.alpha, self.threshold = alpha, None

    def fit(self, thetas, failed):
        self.clf.fit_logistic(thetas, failed)
        return self

    def calibrate(self, failed_thetas):
        scores = np.sort(self.clf(failed_thetas))
        k = int(np.floor((len(scores) + 1) * self.alpha))
        self.threshold = float(scores[k - 1]) if k >= 1 else -np.inf
        return self

    def passes(self, thetas):
        return self.clf(thetas) < self.threshold
