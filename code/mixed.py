# Maximum-likelihood fit with crossed random effects for events and stations.
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


class Crossed:
    def __init__(self, X, y, g_event, g_station):
        self.y = np.asarray(y, float)
        self.N = len(self.y)
        self.X = None if X is None else np.asarray(X, float)
        self.ev_u, self.ev_i = np.unique(g_event, return_inverse=True)
        self.st_u, self.st_i = np.unique(g_station, return_inverse=True)
        self.ne, self.ns = len(self.ev_u), len(self.st_u)
        q = self.ne + self.ns
        Z = np.zeros((self.N, q))
        Z[np.arange(self.N), self.ev_i] = 1.0
        Z[np.arange(self.N), self.ne + self.st_i] = 1.0
        self.Z = Z
        self.ZtZ = Z.T @ Z
        self.Zty = Z.T @ self.y
        self.ZtX = None if self.X is None else Z.T @ self.X

    def _pieces(self, tau2, s2, e2):
        ginv = np.concatenate([np.full(self.ne, 1.0 / tau2), np.full(self.ns, 1.0 / s2)])
        M = self.ZtZ + np.diag(e2 * ginv)
        cf = cho_factor(M, lower=True)
        logdetM = 2.0 * np.sum(np.log(np.diag(cf[0])))
        logdetG = self.ne * np.log(tau2) + self.ns * np.log(s2)
        q = self.ne + self.ns
        logdetV = (self.N - q) * np.log(e2) + logdetG + logdetM
        return cf, logdetV

    def _vinv_apply(self, cf, e2, A, ZtA):
        return (A - self.Z @ cho_solve(cf, ZtA)) / e2

    def _gls(self, cf, e2):
        if self.X is None:
            return np.zeros(0), None
        VinvX = self._vinv_apply(cf, e2, self.X, self.ZtX)
        A = self.X.T @ VinvX
        b = VinvX.T @ self.y
        beta = np.linalg.solve(A, b)
        return beta, A

    def nll(self, p):
        tau2, s2, e2 = np.exp(2 * np.asarray(p))
        cf, logdetV = self._pieces(tau2, s2, e2)
        beta, _ = self._gls(cf, e2)
        r = self.y if self.X is None else self.y - self.X @ beta
        Ztr = self.Z.T @ r
        quad = (r @ r - Ztr @ cho_solve(cf, Ztr)) / e2
        return 0.5 * (logdetV + quad)

    def fit(self):
        if self.X is not None:
            b0 = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
            s = (self.y - self.X @ b0).std()
        else:
            s = self.y.std()
        x0 = np.log([max(0.5 * s, 1e-3), max(0.4 * s, 1e-3), max(0.6 * s, 1e-3)])
        res = minimize(self.nll, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 6000})
        tau2, s2, e2 = np.exp(2 * res.x)
        cf, _ = self._pieces(tau2, s2, e2)
        beta, A = self._gls(cf, e2)
        r = self.y if self.X is None else self.y - self.X @ beta
        u = cho_solve(cf, self.Z.T @ r)        # BLUPs: (Z'Z + e2 G^-1)^-1 Z' r
        eta, dS = u[:self.ne], u[self.ne:]
        eps = r - eta[self.ev_i] - dS[self.st_i]
        se = np.sqrt(np.diag(np.linalg.inv(A))) if A is not None else np.zeros(0)
        return dict(beta=beta, se=se, tau=np.sqrt(tau2), phiS2S=np.sqrt(s2), phiSS=np.sqrt(e2),
                    phi=np.sqrt(s2 + e2), sigma=np.sqrt(tau2 + s2 + e2), eta=eta, dS=dS, eps=eps,
                    events=self.ev_u, stations=self.st_u, ev_i=self.ev_i, st_i=self.st_i,
                    converged=res.success, nll=res.fun, n=self.N, n_events=self.ne, n_stations=self.ns)


def fit_crossed(X, y, g_event, g_station):
    return Crossed(X, y, g_event, g_station).fit()
