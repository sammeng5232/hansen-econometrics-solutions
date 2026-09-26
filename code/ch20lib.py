import os
import numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')


def ols(y, X, cl=None):
    """OLS with HC3 (leave-one-out) or delete-cluster variance; CV and AIC."""
    y = np.asarray(y, float); X = np.asarray(X, float); n, k = X.shape
    Q, Rr = np.linalg.qr(X)
    XXi = np.linalg.inv(Rr.T @ Rr)
    b = XXi @ X.T @ y; e = y - X @ b
    if cl is None:
        h = np.sum(Q * Q, 1); et = e / (1 - h)
        M = (X * et[:, None]).T @ (X * et[:, None])
    else:
        et = np.zeros(n); S = np.zeros((k, k))
        for g in np.unique(cl):
            ix = np.where(cl == g)[0]; Xg = X[ix]
            Hg = Xg @ XXi @ Xg.T
            etg = np.linalg.solve(np.eye(len(ix)) - Hg, e[ix]); et[ix] = etg
            s = Xg.T @ etg; S += np.outer(s, s)
        M = S
    V = XXi @ M @ XXi
    Ri = np.linalg.inv(Rr); Mq = Ri.T @ M @ Ri
    s2 = e @ e / n
    return dict(b=b, V=V, Ri=Ri, Mq=Mq, e=e, et=et, cv=np.mean(et ** 2), aic=n * np.log(s2) + 2 * k, n=n, k=k)


def pred(r, Xg):
    m = Xg @ r['b']; G = Xg @ r['Ri']
    w, U = np.linalg.eigh(r['Mq']); L = U * np.sqrt(np.clip(w, 0, None))
    s = np.sqrt(np.sum((G @ L) ** 2, 1)); return m, s


def poly(x, p):
    return np.column_stack([x ** j for j in range(p + 1)])


def qspline(x, knots):
    return np.column_stack([np.ones_like(x), x, x ** 2] + [((x - t) ** 2) * (x >= t) for t in knots])


def lspline(x, knots):
    return np.column_stack([x] + [(x - t) * (x >= t) for t in knots])

