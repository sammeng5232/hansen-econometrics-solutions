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


out = {}
# ---------------- CPS
c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
lw = np.log(c[4] / (c[5] * c[6])).values; edu = c[3].values.astype(float); exp_ = (c[0] - c[3] - 6).values.astype(float)
print('n', len(lw), 'exp range', exp_.min(), exp_.max(), 'edu range', edu.min(), edu.max(), 'n exp>65', (exp_ > 65).sum())
for name, x in [('exp', exp_), ('edu', edu)]:
    lo, hi = x.min(), x.max(); xs = (x - lo) / (hi - lo)
    rows = []
    for p in range(1, 9):
        r = ols(lw, poly(xs, p)); rows.append((p, r['cv'], r['aic']))
        out[(name, 'poly', p)] = r
    print(name, 'poly CV/AIC', [(p, round(cv, 5), round(a, 1)) for p, cv, a in rows])
    pcv = min(rows, key=lambda t: t[1])[0]; paic = min(rows, key=lambda t: t[2])[0]; print(name, 'CV order', pcv, 'AIC order', paic)
    out[(name, 'pcv')] = pcv; out[(name, 'paic')] = paic
    grid = np.linspace(lo, hi, 300); gs = (grid - lo) / (hi - lo)
    for p in [6, pcv]:
        m, s = pred(out[(name, 'poly', p)], poly(gs, p)); out[(name, 'fit', p)] = (grid, m, s)
    # counts per value
    vals, cnt = np.unique(x, return_counts=True); out[(name, 'counts')] = (vals, cnt)
# exp: report key values of 6th order poly
for name in ['exp', 'edu']:
    grid, m, s = out[(name, 'fit', 6)]
    pts = [0, 5, 10, 20, 30, 40, 50, 60, 65, 70] if name == 'exp' else [0, 4, 8, 10, 12, 14, 16, 18, 20]
    print(name, 'poly6', [(p, round(float(np.interp(p, grid, m)), 3), round(float(np.interp(p, grid, s)), 3)) for p in pts])
    p = out[(name, 'pcv')]; grid, m, s = out[(name, 'fit', p)]
    print(name, 'polyCV', p, [(q, round(float(np.interp(q, grid, m)), 3), round(float(np.interp(q, grid, s)), 3)) for q in pts])
vals, cnt = out[('exp', 'counts')]; print('exp counts >60', list(zip(vals[vals > 60], cnt[vals > 60])))
# splines
spl = {'exp': [[], [20], [20, 40], [10, 20, 30, 40]], 'edu': [[], [10], [5, 10, 15], [4, 8, 12, 16]]}
for name, x in [('exp', exp_), ('edu', edu)]:
    lo, hi = x.min(), x.max(); grid = np.linspace(lo, hi, 300)
    for kn in spl[name]:
        r = ols(lw, qspline(x, kn)); m, s = pred(r, qspline(grid, kn)); out[(name, 'spl', tuple(kn))] = (r, grid, m, s)
        print(name, 'spline', kn, 'CV %.5f AIC %.1f' % (r['cv'], r['aic']))
    pts = [0, 5, 10, 20, 30, 40, 50, 60, 70] if name == 'exp' else [0, 4, 8, 10, 12, 14, 16, 18, 20]
    for kn in spl[name]:
        r, grid, m, s = out[(name, 'spl', tuple(kn))]
        print(' ', kn, [(q, round(float(np.interp(q, grid, m)), 3), round(float(np.interp(q, grid, s)), 3)) for q in pts])
# figures CPS
for name, lab, fn9, fn13 in [('exp', 'Experience (years)', 'ch20_09.pdf', 'ch20_13.pdf'), ('edu', 'Education (years)', 'ch20_11.pdf', 'ch20_14.pdf')]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, p, t in [(ax[0], 6, '(a) 6th-order polynomial'), (ax[1], out[(name, 'pcv')], '(b) CV-selected order %d' % out[(name, 'pcv')])]:
        grid, m, s = out[(name, 'fit', p)]
        a.fill_between(grid, m - 1.96 * s, m + 1.96 * s, color='0.82', lw=0); a.plot(grid, m, 'k', lw=1.4)
        a.set_title(t, fontsize=10); a.set_xlabel(lab); a.set_ylim(1.5, 4.0)
    ax[0].set_ylabel('Log wage'); plt.tight_layout(); plt.savefig(R + fn9); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    sty = ['k-', 'k--', 'k-.', 'k:']
    for kn, st in zip(spl[name], sty):
        r, grid, m, s = out[(name, 'spl', tuple(kn))]
        ax[0].plot(grid, m, st, lw=1.3, label='knots ' + (', '.join(str(k) for k in kn) if kn else 'none (quadratic)'))
    ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('(a) Quadratic splines', fontsize=10); ax[0].set_xlabel(lab); ax[0].set_ylabel('Log wage')
    best = min(spl[name], key=lambda kn: out[(name, 'spl', tuple(kn))][0]['cv'])
    r, grid, m, s = out[(name, 'spl', tuple(best))]
    ax[1].fill_between(grid, m - 1.96 * s, m + 1.96 * s, color='0.82', lw=0); ax[1].plot(grid, m, 'k', lw=1.4)
    ax[1].set_title('(b) CV-selected spline (knots %s), 95%% bands' % ', '.join(str(k) for k in best), fontsize=10); ax[1].set_xlabel(lab)
    plt.tight_layout(); plt.savefig(R + fn13); plt.close()
    print(name, 'best spline', best)

# ---------------- RR2010
rr = pd.read_stata('RR2010/RR2010.dta').sort_values('year')
Y = rr.gdp.values.astype(float); D = rr.debt.values.astype(float)
y = Y[1:]; y1 = Y[:-1]; d1 = D[:-1]
grid = np.linspace(d1.min(), d1.max(), 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for kn, st in zip([[], [60], [40, 80]], ['k--', 'k-', 'k-.']):
    X = np.column_stack([np.ones_like(y), y1, lspline(d1, kn)]); r = ols(y, X)
    G = np.column_stack([np.ones_like(grid), np.full_like(grid, y1.mean()), lspline(grid, kn)]); m, s = pred(r, G)
    se_b = np.sqrt(np.diag(r['V']))
    print('RR knots', kn, 'b', np.round(r['b'], 4), 'se', np.round(se_b, 4), 'CV %.3f AIC %.2f' % (r['cv'], r['aic']))
    # slopes by segment
    sl = np.cumsum(r['b'][2:]); print('  segment slopes', np.round(sl, 4))
    if kn == [60]:
        a_ = np.array([0, 0, 1, 1.0]); print('  slope above 60: %.4f (%.4f)' % (a_ @ r['b'], np.sqrt(a_ @ r['V'] @ a_)))
        ax[1].fill_between(grid, m - 1.96 * s, m + 1.96 * s, color='0.82', lw=0); ax[1].plot(grid, m, 'k', lw=1.4)
    ax[0].plot(grid, m, st, lw=1.3, label={0: 'linear', 1: 'knot at 60', 2: 'knots at 40, 80'}[len(kn)])
print('n', len(y), 'debt>80 count', (d1 > 80).sum(), 'years', rr.year.values[:-1][d1 > 80])
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('(a) Linear spline estimates of $m(D)$', fontsize=10); ax[0].set_xlabel('Lagged debt/GDP (%)'); ax[0].set_ylabel('GDP growth (%)')
ax[1].set_title('(b) One knot at 60, 95% bands', fontsize=10); ax[1].set_xlabel('Lagged debt/GDP (%)'); [a.plot(d1, np.full_like(d1, a.get_ylim()[0]), '|', color='0.5', ms=6) for a in ax]
plt.tight_layout(); plt.savefig(R + 'ch20_15.pdf'); plt.close()

# ---------------- DDK
d = pd.read_stata('DDK2011/DDK2011.dta').dropna(subset=['totalscore', 'percentile'])
y = d.totalscore.values.astype(float); x = d.percentile.values.astype(float); cl = pd.factorize(d.schoolid)[0]
print('DDK n', len(y), 'schools', cl.max() + 1)
knl = [[], [50], [33, 66], [25, 50, 75], [20, 40, 60, 80]]
grid = np.linspace(0, 100, 201)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
res = []
for kn, st in zip(knl, ['k:', 'k--', 'k-.', 'k-', '0.5']):
    r = ols(y, qspline(x, kn), cl=cl); m, s = pred(r, qspline(grid, kn)); res.append((kn, r, m, s))
    r0 = ols(y, qspline(x, kn))
    print('DDK knots', kn, 'cluster CV %.4f  conventional CV %.4f AIC %.1f' % (r['cv'], r0['cv'], r0['aic']))
    ax[0].plot(grid, m, st if not st.startswith('0') else '-', color=None if not st.startswith('0') else '0.5', lw=1.2, label='knots ' + (', '.join(map(str, kn)) if kn else 'none'))
best = min(res, key=lambda t: t[1]['cv'])
kn, r, m, s = best; print('DDK best', kn)
for q in [0, 10, 25, 50, 75, 90, 100]:
    print('  ', q, round(float(np.interp(q, grid, m)), 3), round(float(np.interp(q, grid, s)), 3))
ax[1].fill_between(grid, m - 1.96 * s, m + 1.96 * s, color='0.82', lw=0); ax[1].plot(grid, m, 'k', lw=1.4)
bl = np.polyfit(x, y, 1); ax[1].plot(grid, np.polyval(bl, grid), 'k--', lw=1, label='linear')
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title('(a) Quadratic splines', fontsize=10); ax[0].set_xlabel('Initial percentile'); ax[0].set_ylabel('Test score')
ax[1].set_title('(b) Selected spline (knots %s), clustered 95%% bands' % ', '.join(map(str, kn)), fontsize=10); ax[1].set_xlabel('Initial percentile'); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(R + 'ch20_16.pdf'); plt.close()
# derivative of selected spline
