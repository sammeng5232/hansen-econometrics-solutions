"""Sparse dynamic-panel GMM (Arellano-Bond difference GMM, Blundell-Bond system GMM).

Stata xtdpd conventions (validated: reproduces Blundell-Bond 1998 Table 4 col 3 exactly and Hansen Table 17.3
to within 0.001):
  * GMM-style instruments, one column per (variable, period, lag); missing -> 0.
  * Time effects as year dummies (differenced in the difference equation, levels in the level equation),
    sharing one IV-style column per year across both equations; constant in the level equation.
  * One-step weight (sum Z_i' H Z_i)^+ with H = diag(H_diff, I); two-step weight (sum Z_i'u_i u_i'Z_i)^+.
  * Robust one-step: sandwich; robust two-step: Windmeijer (2005) correction.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _grid(df, idv, tv, cols):
    years = np.arange(int(df[tv].min()), int(df[tv].max()) + 1)
    ids = np.sort(df[idv].unique())
    idx = pd.MultiIndex.from_product([ids, years], names=[idv, tv])
    g = df.set_index([idv, tv])[cols].astype(float)
    g = g[~g.index.duplicated()].reindex(idx)
    return ids, years, {c: g[c].values.reshape(len(ids), len(years)) for c in cols}


def _shift(a, s):
    out = np.full_like(a, np.nan)
    if s == 0:
        return a.copy()
    out[:, s:] = a[:, :-s]
    return out


def _keep_cols(X):
    keep, Q = [], np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        r = v - Q @ (Q.T @ v)
        if np.linalg.norm(r) > 1e-8 * nv:
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
            keep.append(j)
    return keep


def ginv(O, tol=1e-10):
    """Scale-invariant generalized inverse: pinv of the correlation-scaled matrix with relative tolerance."""
    d = np.sqrt(np.clip(np.diag(O), 0, None))
    d[d == 0] = 1.0
    C = O / np.outer(d, d)
    return np.linalg.pinv(C, rcond=tol, hermitian=True) / np.outer(d, d)


def dpd(df, idv, tv, y, regs, dgmm, system=False, lgmm=None, iv_exog=None, timedum=True, twostep=False):
    """regs: list of (var, lag); dgmm: list of (var, lo, hi) (hi=None: all available lags);
    lgmm: list of (var, lag) -> Delta var_{t-lag} for the level equation (default lag lo-1);
    iv_exog: list of (var, lag) strictly exogenous regressors used as IV-style instruments."""
    iv_exog = iv_exog or []
    if system and lgmm is None:
        lgmm = [(v, lo - 1) for v, lo, _ in dgmm]
    lgmm = lgmm or []
    cols = sorted({y, *[v for v, _ in regs], *[v for v, _, _ in dgmm], *[v for v, _ in lgmm], *[v for v, _ in iv_exog]})
    ids, years, A = _grid(df, idv, tv, cols)
    N, T = len(ids), len(years)
    lev = lambda v, s: _shift(A[v], s)
    dif = lambda v, s: _shift(A[v], s) - _shift(A[v], s + 1)
    dy = dif(y, 0); dX = [dif(v, s) for v, s in regs]
    okd = ~np.isnan(dy)
    for a in dX + [dif(v, s) for v, s in iv_exog]:
        okd &= ~np.isnan(a)
    if system:
        ly = lev(y, 0); lX = [lev(v, s) for v, s in regs]
        okl = ~np.isnan(ly)
        for a in lX + [lev(v, s) for v, s in iv_exog]:
            okl &= ~np.isnan(a)
    else:
        okl = np.zeros_like(okd)
    # rows
    di, dt = np.nonzero(okd)
    li, lt = np.nonzero(okl)
    nd, nl = len(di), len(li)
    n = nd + nl
    rid = np.concatenate([di, li]); rt = np.concatenate([dt, lt])
    isd = np.r_[np.ones(nd, bool), np.zeros(nl, bool)]
    # ---- instrument columns
    zr, zc, zv = [], [], []
    ncol = 0
    tper = np.unique(dt)
    for v, lo, hi in dgmm:
        colmap = {}
        for t in tper:
            smax = t if hi is None else min(hi, t)
            for s in range(lo, smax + 1):
                colmap[(t, s)] = ncol; ncol += 1
        a = A[v]
        for s in range(lo, (T if hi is None else hi) + 1):
            sel = dt - s >= 0
            if hi is None:
                pass
            rows = np.nonzero(sel)[0]
            if len(rows) == 0:
                continue
            vals = a[di[rows], dt[rows] - s]
            good = ~np.isnan(vals) & (vals != 0)
            rows, vals = rows[good], vals[good]
            cc = np.array([colmap.get((t, s), -1) for t in dt[rows]])
            ok = cc >= 0
            zr.append(rows[ok]); zc.append(cc[ok]); zv.append(vals[ok])
    tperl = np.unique(lt) if system else []
    for v, s in lgmm:
        colmap = {t: ncol + j for j, t in enumerate(tperl)}; ncol += len(tperl)
        vals = dif(v, s)[li, lt]
        good = ~np.isnan(vals) & (vals != 0)
        rows = nd + np.nonzero(good)[0]
        zr.append(rows); zc.append(np.array([colmap[t] for t in lt[good]])); zv.append(vals[good])
    # IV-style: year dummies (shared column), exogenous regressors (shared column), constant (levels)
    Dm = np.zeros((n, T))
    if timedum:
        Dm[np.arange(nd), dt] += 1
        prev = dt - 1
        ok = prev >= 0
        Dm[np.arange(nd)[ok], prev[ok]] -= 1
        Dm[nd + np.arange(nl), lt] = 1
    Ex = np.zeros((n, len(iv_exog)))
    for j, (v, s) in enumerate(iv_exog):
        Ex[:nd, j] = dif(v, s)[di, dt]
        if system:
            Ex[nd:, j] = lev(v, s)[li, lt]
    cons = np.r_[np.zeros(nd), np.ones(nl)]
    ivblock = np.column_stack([Dm, Ex] + ([cons] if system else []))
    ivkeep = [j for j in range(ivblock.shape[1]) if np.abs(ivblock[:, j]).sum() > 0]
    ivblock = ivblock[:, ivkeep]
    r_, c_ = np.nonzero(ivblock)
    zr.append(r_); zc.append(ncol + c_); zv.append(ivblock[r_, c_])
    ncol += ivblock.shape[1]
    Z = sp.csr_matrix((np.concatenate(zv), (np.concatenate(zr), np.concatenate(zc))), shape=(n, ncol))
    nzc = np.unique(Z.indices)
    Z = Z[:, nzc]
    # ---- regressors
    Xr = np.zeros((n, len(regs)))
    for j, (v, s) in enumerate(regs):
        Xr[:nd, j] = dX[j][di, dt]
        if system:
            Xr[nd:, j] = lX[j][li, lt]
    Yv = np.r_[dy[di, dt], ly[li, lt] if system else np.zeros(0)]
    Xfull = np.column_stack([Xr, Dm] + ([cons] if system else []))
    keep = _keep_cols(Xfull)
    assert keep[:len(regs)] == list(range(len(regs))), 'regressors collinear'
    X = Xfull[:, keep]
    k = X.shape[1]
    # ---- firm map and H
    F = sp.csr_matrix((np.ones(n), (rid, np.arange(n))), shape=(N, n))
    order = np.lexsort((rt, ~isd, rid))  # diff rows sorted by firm, time
    hr, hc, hv = list(range(n)), list(range(n)), list(np.where(isd, 2.0, 1.0))
    dsort = np.lexsort((dt, di))
    a1, a2 = dsort[:-1], dsort[1:]
    adj = (di[a1] == di[a2]) & (dt[a2] - dt[a1] == 1)
    hr += list(a1[adj]) + list(a2[adj]); hc += list(a2[adj]) + list(a1[adj]); hv += [-1.0] * (2 * adj.sum())
    H = sp.csr_matrix((hv, (hr, hc)), shape=(n, n))
    ZHZ = (Z.T @ (H @ Z)).toarray()
    Sxz = (Z.T @ X).T          # k x l
    Szy = Z.T @ Yv
    def solve(W):
        M = Sxz @ W @ Sxz.T
        return np.linalg.solve(M, Sxz @ W @ Szy), M
    def ZU(u):
        return (F @ Z.multiply(u[:, None])).toarray() if False else np.asarray((F @ sp.diags(u) @ Z).todense())
    W1 = ginv(ZHZ)
    th1, M1 = solve(W1)
    u1 = Yv - X @ th1
    G1 = ZU(u1); O1 = G1.T @ G1
    M1i = np.linalg.inv(M1)
    V1 = M1i @ (Sxz @ W1 @ O1 @ W1 @ Sxz.T) @ M1i
    s2 = (u1[:nd] @ u1[:nd]) / (2 * nd)
    out = dict(names=[f'{v}_L{s}' for v, s in regs], nreg=len(regs), nd=nd, nl=nl, N=int((np.asarray(F.sum(1)).ravel() > 0).sum()),
               L=Z.shape[1], th1=th1, V1=V1, se1=np.sqrt(np.diag(V1)), se1c=np.sqrt(np.diag(s2 * M1i)))
    if twostep:
        W2 = ginv(O1)
        th2, M2 = solve(W2)
        M2i = np.linalg.inv(M2)
        u2 = Yv - X @ th2
        g2 = Z.T @ u2
        v = W2 @ g2
        a = G1 @ v
        D = np.zeros((k, k))
        for j in range(k):
            Pk = F @ sp.diags(X[:, j]) @ Z
            t1 = Pk.T @ a
            t2 = G1.T @ (Pk @ v)
            D[:, j] = M2i @ Sxz @ W2 @ (t1 + t2)
        Vw = M2i + D @ M2i + M2i @ D.T + D @ V1 @ D.T
        G2 = ZU(u2)
        J = g2 @ ginv(G2.T @ G2) @ g2
        out.update(th2=th2, V2=Vw, se2=np.sqrt(np.diag(Vw)), se2c=np.sqrt(np.diag(M2i)), J=J)
    return out


def show(r, step=1, digits=4):
    th = r['th%d' % step]; se = r['se%d' % step]
    return '  '.join(f"{nm}={th[j]:.{digits}f}({se[j]:.{digits}f})" for j, nm in enumerate(r['names']))
