import os
import numpy as np, pandas as pd, time, sys
import scipy.sparse as sp
from scipy.optimize import linprog
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')
TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]


def qreg(y, X, tau):
    """Quantile regression via the dual LP (Koenker-Bassett): max y'a s.t. X'a=(1-tau)X'1, 0<=a<=1;
    the coefficients are the dual values of the equality constraints."""
    n, k = X.shape
    r = linprog(-y, A_eq=X.T, b_eq=(1 - tau) * X.sum(0), bounds=[(0, 1)] * n, method='highs')
    return -r.eqlin.marginals


def boot(y, X, tau, B, cl=None, seed=0):
    rng = np.random.default_rng(seed); n = len(y); out = []
    if cl is not None:
        groups = [np.where(cl == g)[0] for g in np.unique(cl)]
    for b in range(B):
        if cl is None:
            idx = rng.integers(0, n, n)
        else:
            gi = rng.integers(0, len(groups), len(groups)); idx = np.concatenate([groups[j] for j in gi])
        out.append(qreg(y[idx], X[idx], tau))
    return np.array(out)


def bc_ci(draws, est, alpha=0.05):
    from scipy.stats import norm
    z0 = norm.ppf(np.mean(draws < est))
    lo = norm.cdf(2 * z0 + norm.ppf(alpha / 2)); hi = norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2))
    return np.quantile(draws, lo), np.quantile(draws, hi)


part = sys.argv[1] if len(sys.argv) > 1 else 'all'
B = 1000
if part in ('rep', 'all'):
    d = pd.read_stata('DDK2011/DDK2011.dta').dropna(subset=['totalscore'])
    ts = (d.totalscore - d.totalscore.mean()) / d.totalscore.std()
    X = np.column_stack([np.ones(len(d)), d.tracking.values.astype(float)])
    print('Table 24.2 replication:', [round(qreg(ts.values, X, t)[1], 3) for t in TAUS], flush=True)
if part in ('cps', 'all'):
    c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
    c['lw'] = np.log(c[4] / (c[5] * c[6]))
    for lab, f in [('men', 0), ('women', 1)]:
        s = c[(c[2] == 1) & (c[1] == f) & (c[3] >= 11)]
        y = s.lw.values; X = np.column_stack([np.ones(len(s)), s[3].values.astype(float)])
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        print('Hispanic', lab, 'n', len(y), 'OLS slope %.4f' % ols[1], 'edu values', np.unique(s[3]).tolist(), flush=True)
        for t in TAUS:
            t0 = time.time(); b = qreg(y, X, t); bd = boot(y, X, t, 500, seed=1)
            se = bd.std(0); lo, hi = bc_ci(bd[:, 1], b[1])
            print('  tau %.1f  const %.3f (%.3f)  edu %.4f (%.4f)  BC [%.4f, %.4f]  %.0fs' % (t, b[0], se[0], b[1], se[1], lo, hi, time.time() - t0), flush=True)
if part in ('ddk', 'all'):
    d = pd.read_stata('DDK2011/DDK2011.dta').dropna(subset=['totalscore', 'percentile'])
    d = d[d.tracking == 1]
    y = d.totalscore.values.astype(float); X = np.column_stack([np.ones(len(d)), d.percentile.values.astype(float)]); cl = d.schoolid.values
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    print('DDK tracked n', len(y), 'schools', len(np.unique(cl)), 'OLS', np.round(ols, 4), flush=True)
    res = []
    for t in TAUS:
        t0 = time.time(); b = qreg(y, X, t); bd = boot(y, X, t, B, cl=cl, seed=2)
        se = bd.std(0); lo, hi = bc_ci(bd[:, 1], b[1]); res.append((t, b, bd))
        print('  tau %.1f  const %.3f (%.3f)  pct %.4f (%.4f)  BC [%.4f, %.4f]  %.0fs' % (t, b[0], se[0], b[1], se[1], lo, hi, time.time() - t0), flush=True)
    # test equality of slopes 0.1 vs 0.9 via bootstrap of the difference (same resamples: use common seed)
    diffs = res[-1][2][:, 1] - res[0][2][:, 1]
    print('  slope(0.9)-slope(0.1) = %.4f, bootstrap se (independent draws approx) %.4f' % (res[-1][1][1] - res[0][1][1], np.sqrt(res[-1][2][:, 1].var() + res[0][2][:, 1].var())))
if part in ('poly', 'all'):
    c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
    c['lw'] = np.log(c[4] / (c[5] * c[6])); c['ex'] = (c[0] - c[3] - 6) / 40
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    xg = np.arange(0, 45.5, 0.5); G = np.column_stack([(xg / 40) ** j for j in range(6)])
    for a, lab, race in [(ax[0], 'Black', 2), (ax[1], 'white', 1)]:
        s = c[(c[3] == 16) & (c[1] == 1) & (c[10] == race)]
        y = s.lw.values; X = np.column_stack([s.ex.values ** j for j in range(6)])
        print('college', lab, 'women n', len(y), 'exp range', (s.ex * 40).min(), (s.ex * 40).max(), 'share exp>35', np.mean(s.ex * 40 > 35), flush=True)
        for t, st in zip(TAUS, ['-', '--', '-', '--', '-']):
            b = qreg(y, X, t); f = G @ b
            a.plot(xg, f, 'k' + st, lw=1.2 if t != 0.5 else 1.8, label=r'$q_{%.1f}$' % t)
            print('  tau %.1f' % t, ' '.join('%d:%.2f' % (e, f[int(e * 2)]) for e in [0, 5, 10, 20, 30, 40]), flush=True)
        a.set_title('College-educated %s women (n = %d)' % (lab, len(y)), fontsize=10); a.set_xlabel('Experience (years)'); a.set_xlim(0, 45)
    ax[0].set_ylabel('Log wage'); ax[0].legend(frameon=False, fontsize=8, ncol=2); ax[1].set_ylim(1.5, 4.5)
    plt.tight_layout(); plt.savefig(R + 'ch24_16.pdf'); plt.close()
