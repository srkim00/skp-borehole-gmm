# Transfer functions of layered media for oblique SH and P-SV incidence.
import numpy as np

def density(vs):
    return np.where(vs <= 180, 1700.0, np.where(vs <= 360, 2000.0, np.where(vs <= 1500, 2200.0, 2500.0)))


def read_models(path):
    lines = open(path).read().split('\n'); i = 0; models = []
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        blk = np.array([[float(x) for x in lines[i + 1 + k].split()] for k in range(n)])
        models.append(blk); i += 1 + n
    return models


def _vert(slow2, p):
    q = np.sqrt(slow2 - p * p + 0j)
    return np.where(q.imag < 0, -q, q)


def layer_props(model, qfac):
    h = model[:-1, 0].astype(float); vs = model[:, 2].astype(float); vp = 1.73 * vs; rho = density(vs)
    qs = vs / qfac; qp = 2.0 * qs
    beta = vs * np.sqrt(1 - 1j / qs); alpha = vp * np.sqrt(1 - 1j / qp)
    mu = rho * beta ** 2; lam = rho * alpha ** 2 - 2 * mu
    return h, alpha, beta, rho, mu, lam


def _cols_psv(a, b, mu, lam, p):
    xi = _vert(1 / a ** 2, p); eta = _vert(1 / b ** 2, p)
    waves = [((a * p, a * xi), xi), ((b * eta, -b * p), eta), ((a * p, -a * xi), -xi), ((b * eta, b * p), -eta)]
    E = np.zeros((4, 4), complex)
    for k, ((ux, uz), q) in enumerate(waves):
        E[:, k] = [ux, uz, mu * (q * ux + p * uz), lam * (p * ux + q * uz) + 2 * mu * q * uz]
    return E, xi, eta


def response_psv(model, depths, freqs, p, qfac=100.0):
    h, a, b, rho, mu, lam = layer_props(model, qfac)
    n = len(a); w = 2 * np.pi * np.asarray(freqs); F = len(w)
    Es, xis, etas = zip(*[_cols_psv(a[j], b[j], mu[j], lam[j], p) for j in range(n)])
    neq = 4 * (n - 1) + 2
    A = np.zeros((F, neq, neq), complex); rhs = np.zeros((F, neq), complex)
    col0 = lambda j: 4 * j
    def ph(q, d):
        return np.exp(1j * w * q * d)
    # free surface (layer 0)
    E = Es[0]; eP = ph(xis[0], h[0]); eS = ph(etas[0], h[0])
    for r in (2, 3):
        row = r - 2
        A[:, row, 0] = E[r, 0]; A[:, row, 1] = E[r, 1]; A[:, row, 2] = E[r, 2] * eP; A[:, row, 3] = E[r, 3] * eS
    # interfaces
    for j in range(n - 1):
        r0 = 2 + 4 * j
        E = Es[j]; eP = ph(xis[j], h[j]); eS = ph(etas[j], h[j]); c = col0(j)
        for r in range(4):
            A[:, r0 + r, c] = E[r, 0] * eP; A[:, r0 + r, c + 1] = E[r, 1] * eS; A[:, r0 + r, c + 2] = E[r, 2]; A[:, r0 + r, c + 3] = E[r, 3]
        E2 = Es[j + 1]; c2 = col0(j + 1)
        if j + 1 < n - 1:
            eP2 = ph(xis[j + 1], h[j + 1]); eS2 = ph(etas[j + 1], h[j + 1])
            for r in range(4):
                A[:, r0 + r, c2] = -E2[r, 0]; A[:, r0 + r, c2 + 1] = -E2[r, 1]
                A[:, r0 + r, c2 + 2] = -E2[r, 2] * eP2; A[:, r0 + r, c2 + 3] = -E2[r, 3] * eS2
        else:
            for r in range(4):
                A[:, r0 + r, c2] = -E2[r, 0]; A[:, r0 + r, c2 + 1] = -E2[r, 1]
                rhs[:, r0 + r] = E2[r, 3]          # incident up-going SV of unit amplitude at the top of the half-space
    x = np.linalg.solve(A, rhs[..., None])[..., 0]
    tops = np.concatenate([[0.0], np.cumsum(h)])
    out = np.zeros((len(depths), F), complex)
    for i, z in enumerate(depths):
        j = min(int(np.searchsorted(tops, z, side='right') - 1), n - 1)
        E = Es[j]
        if j < n - 1:
            dz_t = z - tops[j]; dz_b = tops[j + 1] - z; c = col0(j)
            out[i] = (E[0, 0] * x[:, c] * ph(xis[j], dz_t) + E[0, 1] * x[:, c + 1] * ph(etas[j], dz_t)
                      + E[0, 2] * x[:, c + 2] * ph(xis[j], dz_b) + E[0, 3] * x[:, c + 3] * ph(etas[j], dz_b))
        else:
            dz = z - tops[n - 1]; c = col0(n - 1)
            out[i] = (E[0, 0] * x[:, c] * ph(xis[j], dz) + E[0, 1] * x[:, c + 1] * ph(etas[j], dz)
                      + E[0, 3] * 1.0 * ph(etas[j], -dz))
    return out


def response_sh(model, depths, freqs, p, qfac=100.0):
    h, a, b, rho, mu, lam = layer_props(model, qfac)
    n = len(b); w = 2 * np.pi * np.asarray(freqs); F = len(w)
    etas = [_vert(1 / b[j] ** 2, p) for j in range(n)]
    neq = 2 * (n - 1) + 1
    A = np.zeros((F, neq, neq), complex); rhs = np.zeros((F, neq), complex)
    ph = lambda q, d: np.exp(1j * w * q * d)
    t = lambda j, s: mu[j] * s * etas[j]        # traction/(i w) of a wave with q = s*eta and unit displacement
    e0 = ph(etas[0], h[0])
    A[:, 0, 0] = t(0, 1); A[:, 0, 1] = t(0, -1) * e0
    for j in range(n - 1):
        r0 = 1 + 2 * j; c = 2 * j; e = ph(etas[j], h[j])
        A[:, r0, c] = e; A[:, r0, c + 1] = 1.0
        A[:, r0 + 1, c] = t(j, 1) * e; A[:, r0 + 1, c + 1] = t(j, -1)
        c2 = 2 * (j + 1)
        if j + 1 < n - 1:
            e2 = ph(etas[j + 1], h[j + 1])
            A[:, r0, c2] = -1.0; A[:, r0, c2 + 1] = -e2
            A[:, r0 + 1, c2] = -t(j + 1, 1); A[:, r0 + 1, c2 + 1] = -t(j + 1, -1) * e2
        else:
            A[:, r0, c2] = -1.0; A[:, r0 + 1, c2] = -t(j + 1, 1)
            rhs[:, r0] = 1.0; rhs[:, r0 + 1] = t(j + 1, -1)
    x = np.linalg.solve(A, rhs[..., None])[..., 0]
    tops = np.concatenate([[0.0], np.cumsum(h)])
    out = np.zeros((len(depths), F), complex)
    for i, z in enumerate(depths):
        j = min(int(np.searchsorted(tops, z, side='right') - 1), n - 1)
        if j < n - 1:
            c = 2 * j
            out[i] = x[:, c] * ph(etas[j], z - tops[j]) + x[:, c + 1] * ph(etas[j], tops[j + 1] - z)
        else:
            dz = z - tops[n - 1]
            out[i] = x[:, 2 * (n - 1)] * ph(etas[j], dz) + ph(etas[j], -dz)
    return out


def slownesses(model, pmax_crust=0.285e-3, factor=0.985, step=0.005e-3):
    pmax = min(pmax_crust, factor / model[-1, 2])
    return np.arange(0.0, pmax + 1e-12, step)


def tf_oblique(model, z_acc, z_bb, freqs, qfac=100.0):
    ps = slownesses(model)
    s_sv = np.zeros((2, len(freqs))); s_sh = np.zeros((2, len(freqs)))
    for p in ps:
        u = response_psv(model, [z_acc, z_bb], freqs, p, qfac); s_sv += np.abs(u) ** 2
        v = response_sh(model, [z_acc, z_bb], freqs, p, qfac); s_sh += np.abs(v) ** 2
    return np.sqrt(s_sv[1] / s_sv[0]), np.sqrt(s_sh[1] / s_sh[0])


_KO = {}


def ko_smooth(y, freqs, b=40.0):
    key = (len(freqs), float(freqs[0]), float(freqs[-1]), b)
    if key not in _KO:
        lf = np.log10(freqs); x = b * (lf[:, None] - lf[None, :])
        wgt = np.ones_like(x); nz = x != 0
        wgt[nz] = (np.sin(x[nz]) / x[nz]) ** 4
        _KO[key] = wgt / wgt.sum(axis=1, keepdims=True)
    return _KO[key] @ y
