import numpy as np, pandas as pd
from scipy.optimize import minimize, linprog
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')


def ols(y, X, cl=None):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; n, k = X.shape
    XXi = np.linalg.inv(X.T @ X)
    if cl is None:
        M = (X * e[:, None]).T @ (X * e[:, None]) * n / (n - k)
    else:
        u = pd.DataFrame(X * e[:, None]).groupby(cl).sum().values; G = u.shape[0]
        M = u.T @ u * G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(XXi @ M @ XXi))


def tobit(y, X, c, upper=False):
    """censored normal regression; lower censoring at c (y=max(y*,c)) or upper (y=min(y*,c))."""
    cens = (y >= c - 1e-12) if upper else (y <= c + 1e-12)
    s = -1.0 if upper else 1.0
    def nll(t):
        b, ls = t[:-1], t[-1]; sig = np.exp(ls); xb = X @ b
        ll_c = norm.logcdf(s * (c - xb) / sig)
        ll_u = norm.logpdf((y - xb) / sig) - ls
        return -np.sum(np.where(cens, ll_c, ll_u))
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]; t0 = np.r_[b0, np.log(np.std(y))]
    r = minimize(nll, t0, method='BFGS', options={'gtol': 1e-8, 'maxiter': 10000})
    # robust (sandwich) se via numerical derivatives
    th = r.x; k = len(th); h = 1e-5
    def lli(t):
        b, ls = t[:-1], t[-1]; sig = np.exp(ls); xb = X @ b
        return np.where(cens, norm.logcdf(s * (c - xb) / sig), norm.logpdf((y - xb) / sig) - ls)
    G = np.column_stack([(lli(th + h * np.eye(k)[j]) - lli(th - h * np.eye(k)[j])) / (2 * h) for j in range(k)])
    H = np.zeros((k, k))
    for j in range(k):
        e1 = h * np.eye(k)[j]
        gp = np.array([(lli(th + e1 + h * np.eye(k)[m]) - lli(th + e1 - h * np.eye(k)[m])).sum() / (2 * h) for m in range(k)])
        gm = np.array([(lli(th - e1 + h * np.eye(k)[m]) - lli(th - e1 - h * np.eye(k)[m])).sum() / (2 * h) for m in range(k)])
        H[:, j] = (gp - gm) / (2 * h)
    Hi = np.linalg.inv(-H); V = Hi @ (G.T @ G) @ Hi
    se = np.sqrt(np.diag(V))
    return th[:-1], se[:-1], np.exp(th[-1]), cens.mean()


def lad(y, X):
    n, k = X.shape
    r = linprog(-y, A_eq=X.T, b_eq=0.5 * X.sum(0), bounds=[(0, 1)] * n, method='highs')
    return -r.eqlin.marginals


def clad(y, X, c, upper=False, it=100):
    """Buchinsky's iterative LAD for censored median regression."""
    b = lad(y, X)
    for _ in range(it):
        xb = X @ b; keep = (xb < c) if upper else (xb > c)
        b_new = lad(y[keep], X[keep])
        if np.max(np.abs(b_new - b)) < 1e-10: b = b_new; break
        b = b_new
    obj = np.mean(np.abs(y - (np.minimum(X @ b, c) if upper else np.maximum(X @ b, c))))
    return b, obj, keep.mean()


def clad_boot(y, X, c, upper, B=200, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(clad(y[i], X[i], c, upper)[0])
    return np.std(out, axis=0)


def show(lab, b, se, names):
    print('  %-22s' % lab, '  '.join('%s=%.4f(%.4f)' % (nm, bb, ss) for nm, bb, ss in zip(names, b, se)), flush=True)


# ---------------- 27.9 CHJ2004 in-kind transfers
d = pd.read_stata('progs/CHJ2004.dta')
y = d.tinkind.values / 1000; inc = d.income.values / 1000
X = np.column_stack([np.ones_like(inc), inc, (inc - 1) * (inc > 1)])
nm = ['const', 'income', 'Dincome']
print('27.9 n', len(y), 'censored share', np.mean(y == 0), 'mean tinkind', y.mean(), 'income>1 share', np.mean(inc > 1))
b, se = ols(y, X); show('(a) OLS', b, se, nm)
pos = y > 0; b, se = ols(y[pos], X[pos]); show('(c) OLS y>0', b, se, nm)
b, se, sig, cs = tobit(y, X, 0.0); show('(d) Tobit', b, se, nm); print('     sigma %.3f' % sig)
b, obj, frac = clad(y, X, 0.0); seb = clad_boot(y, X, 0.0, False, B=100); show('(e) CLAD', b, seb, nm); print('     frac with xb>0 %.3f' % frac)
# ---------------- 27.10 CPS capped wages
c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c = c[c[3] >= 12]
lw = np.log(c[4] / (c[5] * c[6])).values; e = c[3].values.astype(float)
X = np.column_stack([np.ones_like(e), e, e ** 2]); nm = ['const', 'edu', 'edu2']
cw = np.minimum(lw, 3.4)
print('27.10 n', len(lw), 'share capped', np.mean(lw >= 3.4))
b, se = ols(lw, X); show('(a) OLS lwage', b, se, nm); ra = b
b, se = ols(cw, X); show('(b) OLS cwage', b, se, nm); rb = b
keep = cw < 3.4; b, se = ols(cw[keep], X[keep]); show('(c) OLS cwage<3.4', b, se, nm); rc = b
b, se, sig, cs = tobit(cw, X, 3.4, upper=True); show('(d) Tobit (upper)', b, se, nm); print('     sigma %.3f' % sig); rd = b
cw33 = np.minimum(lw, 3.3)
b, obj, frac = clad(cw33, X, 3.3, upper=True); seb = clad_boot(cw33, X, 3.3, True, B=100); show('(e) CLAD (cap 3.3)', b, seb, nm); re = b
b, obj, frac = clad(cw, X, 3.4, upper=True); print('     CLAD at 3.4:', np.round(b, 4), 'frac xb<c', round(frac, 3))
bm = np.column_stack([np.ones(1), [0]]);
for lab, bb in [('OLS', ra), ('OLS capped', rb), ('truncated', rc), ('Tobit', rd), ('CLAD', re)]:
    print('   %-12s return at 12: %.3f, 16: %.3f, 20: %.3f;  fitted at 12,16,20:' % (lab, bb[1] + 2 * bb[2] * 12, bb[1] + 2 * bb[2] * 16, bb[1] + 2 * bb[2] * 20), np.round([bb[0] + bb[1] * x + bb[2] * x * x for x in (12, 16, 20)], 3))
# ---------------- 27.11 DDK censored at 0
d = pd.read_stata('DDK2011/DDK2011.dta').dropna(subset=['totalscore', 'percentile', 'tracking'])
ts = ((d.totalscore - d.totalscore.mean()) / d.totalscore.std()).values
p = d.percentile.values.astype(float); cl = d.schoolid.values
X = np.column_stack([np.ones_like(p), d.tracking.values.astype(float), p, p ** 2]); nm = ['const', 'tracking', 'pct', 'pct2']
print('27.11 n', len(ts), 'share censored (<0)', np.mean(ts < 0))
b, se = ols(ts, X, cl); show('(a) OLS testscore', b, se, nm)
ct = np.maximum(ts, 0)
b, se = ols(ct, X, cl); show('(b) OLS ctest', b, se, nm)
keep = ct > 0; b, se = ols(ct[keep], X[keep], cl[keep]); show('(c) OLS ctest>0', b, se, nm)
b, se, sig, cs = tobit(ct, X, 0.0); show('    Tobit (extra)', b, se, nm)
