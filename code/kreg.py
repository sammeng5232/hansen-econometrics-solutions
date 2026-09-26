"""Kernel regression (Nadaraya-Watson and local linear) following Hansen (2022) Ch. 19 and his R code:
Gaussian kernel, Fan-Gijbels ROT bandwidth (19.9) from a quartic pilot, leave-one-out CV,
standard errors (19.18) from leave-one-out prediction errors (clustered version (19.25))."""
import numpy as np

phi = lambda u: np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def hrot(x, y, lo, hi, q=4):
    Z = np.column_stack([x ** j for j in range(q + 1)])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ b
    s2 = e @ e / (len(y) - q - 1)
    m2 = sum(j * (j - 1) * b[j] * x ** (j - 2) for j in range(2, q + 1))
    B = np.mean((0.5 * m2) ** 2 * ((x >= lo) & (x <= hi)))
    return 0.58 * ((hi - lo) * s2 / (len(y) * B)) ** 0.2


def _sums(xg, x, w_y, h, mask=None):
    """kernel sums at evaluation points xg (chunked). returns S0,S1,S2,T0,T1 arrays."""
    out = np.zeros((5, len(xg)))
    for a in range(0, len(xg), 800):
        d = x[None, :] - xg[a:a + 800, None]
        K = phi(d / h)
        if mask is not None:
            K = K * mask(a, min(a + 800, len(xg)))
        out[0, a:a + 800] = K.sum(1)
        out[1, a:a + 800] = (K * d).sum(1)
        out[2, a:a + 800] = (K * d * d).sum(1)
        out[3, a:a + 800] = K @ w_y
        out[4, a:a + 800] = (K * d) @ w_y
    return out


def fit(xg, x, y, h, method='LL'):
    S0, S1, S2, T0, T1 = _sums(xg, x, y, h)
    if method == 'NW':
        return T0 / S0
    return (S2 * T0 - S1 * T1) / (S0 * S2 - S1 ** 2)


def loo(x, y, h, method='LL', cl=None):
    """leave-one-out (or leave-cluster-out) predictions at the sample points."""
    if cl is None:
        mask = lambda a, b: 1.0 - (np.arange(a, b)[:, None] == np.arange(len(x))[None, :])
    else:
        mask = lambda a, b: (cl[a:b, None] != cl[None, :]).astype(float)
    S0, S1, S2, T0, T1 = _sums(x, x, y, h, mask)
    if method == 'NW':
        return T0 / S0
    return (S2 * T0 - S1 * T1) / (S0 * S2 - S1 ** 2)


def cv(x, y, hs, method='LL', cl=None, lo=-np.inf, hi=np.inf):
    trim = (x >= lo) & (x <= hi)
    return np.array([np.mean(((y - loo(x, y, h, method, cl)) ** 2)[trim]) for h in hs])


def se(xg, x, e, h, method='LL', cl=None):
    """(19.18) with prediction errors e; clustered (19.25) if cl given."""
    out = np.zeros(len(xg))
    for j, x0 in enumerate(xg):
        d = x - x0
        k = phi(d / h)
        if method == 'NW':
            if cl is None:
                v = np.sum(k * k * e * e) / k.sum() ** 2
            else:
                s = np.bincount(cl, weights=k * e)
                v = np.sum(s * s) / k.sum() ** 2
            out[j] = np.sqrt(v)
        else:
            Z = np.column_stack([np.ones_like(d), d])
            A = np.linalg.inv((Z * k[:, None]).T @ Z)
            ze = Z * (k * e)[:, None]
            if cl is None:
                M = ze.T @ ze
            else:
                R = np.column_stack([np.bincount(cl, weights=ze[:, 0]), np.bincount(cl, weights=ze[:, 1])])
                M = R.T @ R
            out[j] = np.sqrt((A @ M @ A)[0, 0])
    return out
