import numpy as np, pandas as pd
from scipy import stats


def demean(df, cols, g):
    return df[cols] - df.groupby(g)[cols].transform('mean')


def ols_cluster(y, X, cl, names=None, dfk=None, fe_absorbed=False):
    """OLS with cluster-robust variance, Stata small-sample adjustment G/(G-1)*(n-1)/(n-k).
    dfk: number of parameters used in (n-1)/(n-k) (default X.shape[1])."""
    y = np.asarray(y, float); X = np.asarray(X, float); cl = np.asarray(cl)
    n, k = X.shape
    XX = np.linalg.inv(X.T @ X)
    b = XX @ X.T @ y
    e = y - X @ b
    u = pd.DataFrame(X * e[:, None]).groupby(cl).sum().values
    G = u.shape[0]
    kk = k if dfk is None else dfk
    adj = G / (G - 1) * (n - 1) / (n - kk)
    V = adj * XX @ (u.T @ u) @ XX
    return dict(b=b, se=np.sqrt(np.diag(V)), V=V, e=e, n=n, G=G, names=names)


def wald(r, idx):
    b = r['b'][idx]; V = r['V'][np.ix_(idx, idx)]
    W = b @ np.linalg.solve(V, b)
    q = len(idx)
    return W, stats.chi2.sf(W, q), W / q, stats.f.sf(W / q, q, r['G'] - 1)


def xtreg_fe(df, y, xs, g, cl=None):
    """Stata xtreg, fe vce(robust): within regression, cluster by g (or cl), adj G/(G-1)*(n-1)/(n-k-1)."""
    d = df.dropna(subset=[y] + xs).copy()
    W = demean(d, [y] + xs, g)
    X = W[xs].values
    # Stata xtreg,fe reports a constant; df: k regressors + constant
    r = ols_cluster(W[y].values, X, d[cl or g].values, names=xs, dfk=len(xs) + 1)
    r['df'] = d
    return r


def show(r, digits=3):
    return '  '.join(f"{nm}={b:.{digits}f}({s:.{digits}f})" for nm, b, s in zip(r['names'], r['b'], r['se']))
