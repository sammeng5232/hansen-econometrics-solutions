import numpy as np
from scipy.optimize import least_squares

def var_est(Y, p, trend=False, start=None):
    """Y: T x m array. Returns dict with A (list of m x m), c, Sigma (divisor n), resid, n, X design.
    start: index of first dependent observation (>= p) to fix a common sample."""
    T, m = Y.shape
    s = p if start is None else start
    rows = range(s, T)
    Xl = [np.ones(len(rows))]
    if trend: Xl.append(np.arange(s, T, dtype=float))
    for l in range(1, p + 1):
        Xl.append(Y[s - l:T - l])
    X = np.column_stack(Xl)
    Yd = Y[s:]
    B = np.linalg.lstsq(X, Yd, rcond=None)[0]  # (k x m)
    E = Yd - X @ B
    n = len(Yd)
    Sig = E.T @ E / n
    off = 2 if trend else 1
    A = [B[off + (l - 1) * m: off + l * m].T for l in range(1, p + 1)]
    return dict(A=A, c=B[0], B=B, Sigma=Sig, resid=E, n=n, X=X, p=p, m=m, off=off, trend=trend, start=s)

def ma_coefs(A, H):
    m = A[0].shape[0]; p = len(A)
    Th = [np.eye(m)]
    for h in range(1, H + 1):
        M = np.zeros((m, m))
        for l in range(1, min(h, p) + 1):
            M += A[l - 1] @ Th[h - l]
        Th.append(M)
    return Th

def irf(A, Bmat, H, cumulative=None):
    """returns array (H+1, m, m): [h, response i, shock j]. cumulative: list of response indices to cumulate."""
    Th = ma_coefs(A, H)
    R = np.array([T @ Bmat for T in Th])
    if cumulative:
        for i in cumulative:
            R[:, i, :] = np.cumsum(R[:, i, :], axis=0)
    return R

def aic_var(Y, pmax, trend=False):
    out = []
    for p in range(1, pmax + 1):
        v = var_est(Y, p, trend, start=pmax)
        k = v['X'].shape[1] * v['m']
        out.append((p, v['n'], v['n'] * np.log(np.linalg.det(v['Sigma'])) + 2 * k))
    return out

def svar_A(Sigma, template):
    """template: m x m array with floats for fixed entries and np.nan for free entries.
    Solve for free entries so that A Sigma A' is diagonal (just-identified case)."""
    m = Sigma.shape[0]
    free = np.argwhere(np.isnan(template))
    def build(x):
        A = template.copy(); A[tuple(free.T)] = x; return A
    def resid(x):
        A = build(x); M = A @ Sigma @ A.T
        return M[np.triu_indices(m, 1)]
    best = None
    rng = np.random.default_rng(0)
    for trial in range(200):
        x0 = rng.normal(scale=0.5, size=len(free))
        sol = least_squares(resid, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-20:
            break
    A = build(best.x); D = A @ Sigma @ A.T
    return A, np.diag(np.diag(D)), best.cost

def bootstrap_bands(Y, p, Bfun, H, reps, seed, trend=False, cumulative=None, levels=(0.05, 0.95)):
    """recursive-design residual bootstrap percentile bands for IRFs computed by Bfun(v) -> impact matrix."""
    v = var_est(Y, p, trend)
    rng = np.random.default_rng(seed)
    T, m = Y.shape; E = v['resid'] - v['resid'].mean(0)
    draws = []
    for r in range(reps):
        Ys = np.zeros_like(Y); Ys[:p] = Y[:p]
        idx = rng.integers(0, len(E), T - p)
        for t in range(p, T):
            val = v['c'].copy()
            if trend: val = val + v['B'][1] * t
            for l in range(1, p + 1): val = val + v['A'][l - 1] @ Ys[t - l]
            Ys[t] = val + E[idx[t - p]]
        vs = var_est(Ys, p, trend)
        draws.append(irf(vs['A'], Bfun(vs), H, cumulative))
    D = np.array(draws)
    return np.quantile(D, levels[0], axis=0), np.quantile(D, levels[1], axis=0)
