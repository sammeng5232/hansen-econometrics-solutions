import numpy as np

# Table 16.1 (Hansen 2022)
PCT = np.array([0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99])
ADF_CV = {
    'none': np.array([-3.92, -3.28, -2.56, -2.31, -2.15, -2.03, -1.94, -1.79, -1.62, -1.40, -1.23, -0.96, -0.50, 0.05, 0.89, 2.02]),
    'c': np.array([-4.69, -4.08, -3.43, -3.20, -3.06, -2.95, -2.86, -2.72, -2.57, -2.37, -2.22, -1.97, -1.57, -1.15, -0.44, 0.60]),
    'ct': np.array([-5.21, -4.58, -3.95, -3.73, -3.60, -3.50, -3.41, -3.28, -3.13, -2.94, -2.79, -2.56, -2.18, -1.81, -1.24, -0.32]),
}
KPSS_CV = {
    'c': np.array([1.598, 1.176, 0.744, 0.621, 0.550, 0.500, 0.462, 0.406, 0.348, 0.284, 0.241, 0.185, 0.119, 0.079, 0.046, 0.025]),
    'ct': np.array([0.430, 0.324, 0.218, 0.187, 0.169, 0.157, 0.148, 0.134, 0.119, 0.103, 0.091, 0.076, 0.056, 0.041, 0.028, 0.017]),
}
# Tables 16.6-16.8: rows = PCT[:-1] (0.01%..90%), cols m-r = 1..12
JPCT = PCT[:-1]
JOH = {
    2: np.array([
        [22.4, 37.3, 55.7, 78.5, 105, 135, 169, 208, 250, 296, 347, 402],
        [17.6, 31.5, 48.8, 70.1, 95.7, 125, 158, 196, 237, 282, 332, 385],
        [12.8, 25.1, 41.3, 61.3, 85.4, 113, 146, 182, 222, 266, 314, 366],
        [11.3, 23.1, 38.7, 58.4, 81.9, 110, 141, 177, 216, 260, 308, 359],
        [10.4, 21.9, 37.2, 56.5, 79.8, 107, 138, 174, 213, 256, 304, 355],
        [9.71, 21.0, 36.1, 55.2, 78.3, 105, 136, 171, 210, 254, 301, 352],
        [9.19, 20.3, 35.2, 54.1, 77.0, 104, 135, 170, 208, 251, 298, 349],
        [8.42, 19.2, 33.8, 52.5, 75.0, 102, 132, 167, 205, 248, 295, 345],
        [7.57, 18.0, 32.3, 50.6, 72.8, 99.0, 129, 163, 202, 244, 290, 341],
        [6.60, 16.6, 30.4, 48.3, 70.1, 95.9, 126, 159, 197, 239, 285, 335],
        [5.89, 15.5, 29.0, 46.5, 67.9, 93.4, 123, 156, 194, 235, 281, 330],
        [4.86, 13.9, 26.8, 43.7, 64.6, 89.5, 119, 151, 188, 229, 274, 323],
        [3.45, 11.4, 23.4, 39.4, 59.4, 83.4, 111, 143, 179, 219, 263, 312],
        [2.39, 9.39, 20.4, 35.5, 54.6, 77.6, 105, 136, 171, 210, 253, 300],
        [1.35, 6.96, 16.7, 30.4, 48.1, 69.9, 95.7, 125, 159, 197, 239, 285]]),
    3: np.array([
        [15.2, 31.5, 49.0, 71.0, 96.9, 125, 159, 196, 238, 283, 333, 386],
        [10.8, 25.9, 42.8, 63.3, 87.7, 116, 148, 185, 225, 269, 318, 370],
        [6.63, 20.0, 35.5, 54.7, 77.9, 105, 136, 171, 210, 253, 300, 351],
        [5.42, 18.1, 33.2, 51.9, 74.5, 101, 132, 166, 205, 247, 294, 345],
        [4.72, 17.0, 31.7, 50.2, 72.5, 98.9, 128, 163, 202, 244, 290, 341],
        [4.23, 16.2, 30.7, 48.9, 71.0, 97.1, 127, 161, 199, 241, 288, 338],
        [3.85, 15.5, 29.8, 47.9, 69.8, 95.7, 126, 160, 197, 239, 285, 335],
        [3.29, 14.5, 28.5, 46.3, 67.9, 93.6, 123, 157, 194, 236, 282, 331],
        [2.71, 13.4, 27.1, 44.5, 65.8, 91.1, 120, 154, 191, 232, 277, 327],
        [2.08, 12.2, 25.3, 42.3, 63.2, 88.1, 117, 150, 187, 227, 272, 321],
        [1.64, 11.2, 24.0, 40.7, 61.2, 85.7, 114, 147, 183, 224, 268, 317],
        [1.07, 9.75, 22.0, 38.0, 58.0, 82.0, 110, 142, 178, 218, 262, 310],
        [0.45, 7.68, 18.9, 34.0, 53.1, 76.2, 103, 134, 169, 208, 251, 298],
        [0.15, 5.96, 16.2, 30.4, 48.5, 70.7, 96.8, 127, 161, 199, 241, 287],
        [0.02, 4.04, 12.8, 25.7, 42.5, 63.3, 88.1, 117, 150, 187, 227, 272]]),
    4: np.array([
        [27.4, 44.4, 64.6, 90.0, 117, 150, 186, 226, 271, 319, 372, 428],
        [22.1, 38.1, 57.4, 81.0, 108, 139, 175, 214, 258, 305, 356, 412],
        [16.6, 31.2, 49.4, 71.5, 97.6, 128, 162, 200, 242, 288, 338, 392],
        [14.9, 29.0, 46.7, 68.4, 94.0, 124, 157, 195, 236, 282, 332, 385],
        [13.9, 27.6, 45.1, 66.4, 91.8, 121, 154, 192, 233, 278, 328, 381],
        [13.1, 26.7, 43.9, 65.0, 90.1, 119, 152, 189, 230, 275, 325, 378],
        [12.5, 25.9, 42.9, 63.9, 88.8, 118, 151, 187, 228, 273, 322, 375],
        [11.7, 24.7, 41.4, 62.1, 86.7, 115, 148, 184, 225, 270, 318, 371],
        [10.7, 23.3, 39.8, 60.1, 84.4, 113, 145, 181, 221, 266, 314, 366],
        [9.53, 21.7, 37.7, 57.6, 81.5, 109, 141, 177, 217, 261, 309, 360],
        [8.70, 20.5, 36.2, 55.7, 79.2, 107, 138, 174, 213, 257, 304, 356],
        [7.45, 18.7, 33.8, 52.8, 75.7, 103, 134, 169, 207, 250, 297, 348],
        [5.70, 15.9, 30.0, 48.1, 70.2, 96.2, 126, 160, 198, 240, 286, 336],
        [4.28, 13.5, 26.7, 43.8, 65.0, 90.1, 119, 152, 189, 231, 276, 325],
        [2.79, 10.5, 22.4, 38.2, 58.0, 81.8, 110, 141, 177, 217, 261, 309]]),
}


def interp_p(stat, cv, pct, lower_tail=True):
    """Linear interpolation of asymptotic p-value from a critical value table (as in Hansen Sec 16.13)."""
    if lower_tail:
        # cv increasing with pct
        if stat <= cv[0]: return ('<', pct[0])
        if stat >= cv[-1]: return ('>', pct[-1])
        return ('=', float(np.interp(stat, cv, pct)))
    else:
        # cv decreasing with pct
        if stat >= cv[0]: return ('<', pct[0])
        if stat <= cv[-1]: return ('>', pct[-1])
        return ('=', float(np.interp(-stat, -cv, pct)))


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n, k = X.shape
    s2 = e @ e / (n - k)
    V = s2 * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(V)), e


def adf(y, p, det='ct', start=None):
    """ADF regression: dY_t on [const, trend], Y_{t-1}, dY_{t-1..t-p+1}. p = AR order in levels.
    start: first index t of dependent obs (default p)."""
    y = np.asarray(y, float); T = len(y)
    s = p if start is None else start
    t = np.arange(s, T)
    dy = y[t] - y[t - 1]
    cols = []
    if det in ('c', 'ct'): cols.append(np.ones(len(t)))
    if det == 'ct': cols.append(t.astype(float))
    cols.append(y[t - 1])
    for j in range(1, p):
        cols.append(y[t - j] - y[t - j - 1])
    X = np.column_stack(cols)
    b, se, e = ols(dy, X)
    i = {'none': 0, 'c': 1, 'ct': 2}[det]
    return dict(rho=b[i], se=se[i], t=b[i] / se[i], n=len(t), p=p)


def aic_ar(y, pmax, det='ct', start=None):
    """AIC (Stata convention -2logL+2k, Gaussian) for AR(p) in levels with det terms on a common sample."""
    y = np.asarray(y, float); T = len(y)
    s = pmax if start is None else start
    out = []
    for p in range(1, pmax + 1):
        t = np.arange(s, T)
        cols = []
        if det in ('c', 'ct'): cols.append(np.ones(len(t)))
        if det == 'ct': cols.append(t.astype(float))
        for j in range(1, p + 1): cols.append(y[t - j])
        X = np.column_stack(cols)
        b = np.linalg.lstsq(X, y[t], rcond=None)[0]
        e = y[t] - X @ b; n = len(t); k = X.shape[1] + 1
        ll = -n / 2 * (np.log(2 * np.pi) + np.log(e @ e / n) + 1)
        out.append(-2 * ll + 2 * k)
    return int(np.argmin(out)) + 1, out


def kpss(y, M, det='ct'):
    y = np.asarray(y, float); n = len(y)
    if det == 'c':
        e = y - y.mean()
    else:
        X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    w2 = e @ e / n
    for l in range(1, M + 1):
        w2 += 2 * (1 - l / (M + 1)) * (e[l:] @ e[:-l]) / n
    S = np.cumsum(e)
    return (S @ S) / (n ** 2 * w2)


def johansen(Y, p, model=2):
    """Johansen trace statistics for VAR(p) in levels (VECM with p-1 lagged differences).
    model 2: restricted constant; 3: unrestricted constant; 4: restricted trend + unrestricted constant."""
    Y = np.asarray(Y, float); T, m = Y.shape
    t = np.arange(p, T)
    n = len(t)
    Z0 = Y[t] - Y[t - 1]
    Z1 = [Y[t - 1]]
    Z2 = [Y[t - j] - Y[t - j - 1] for j in range(1, p)]
    if model == 2:
        Z1.append(np.ones((n, 1)))
    elif model == 3:
        Z2.append(np.ones((n, 1)))
    elif model == 4:
        Z1.append(t.reshape(-1, 1).astype(float))
        Z2.append(np.ones((n, 1)))
    Z1 = np.column_stack(Z1)
    if Z2:
        Z2 = np.column_stack(Z2)
        P = Z2 @ np.linalg.pinv(Z2)
        R0 = Z0 - P @ Z0; R1 = Z1 - P @ Z1
    else:
        R0, R1 = Z0, Z1
    S00 = R0.T @ R0 / n; S01 = R0.T @ R1 / n; S11 = R1.T @ R1 / n
    Mx = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    lam, V = np.linalg.eig(Mx)
    idx = np.argsort(-lam.real)
    lam = lam.real[idx]; V = V.real[:, idx]
    lam_m = lam[:m]
    LR = [-n * np.sum(np.log(1 - lam_m[r:])) for r in range(m)]
    beta = V[:, 0] / V[0, 0]
    return dict(LR=LR, lam=lam_m, beta=beta, n=n)


def aic_var_levels(Y, pmax, trend=False, start=None):
    Y = np.asarray(Y, float); T, m = Y.shape
    s = pmax if start is None else start
    out = []
    for p in range(1, pmax + 1):
        t = np.arange(s, T)
        cols = [np.ones(len(t))]
        if trend: cols.append(t.astype(float))
        for j in range(1, p + 1): cols.append(Y[t - j])
        X = np.column_stack(cols)
        B = np.linalg.lstsq(X, Y[t], rcond=None)[0]
        E = Y[t] - X @ B; n = len(t)
        S = E.T @ E / n
        out.append(n * np.log(np.linalg.det(S)) + 2 * X.shape[1] * m)
    return int(np.argmin(out)) + 1, out


def fmt_p(res):
    s, v = res
    if s == '<': return '<%.3g' % v
    if s == '>': return '>%.2f' % v
    return '%.2f' % v
