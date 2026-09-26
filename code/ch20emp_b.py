import os
import numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import importlib.util, sys
spec = importlib.util.spec_from_file_location('base', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch20lib.py')); base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
ols, pred, poly = base.ols, base.pred, base.poly
R = base.R

# ---------------- CHJ2004
d = pd.read_stata('progs/CHJ2004.dta')
y = d.transfers.values.astype(float); inc = d.income.values.astype(float)
ctrl_full = ['primary', 'somesecondary', 'secondary', 'someuniversity', 'university', 'age', 'female', 'married',
             'child1', 'child7', 'child15', 'size', 'bothwork', 'notemployed', 'marriedf']
ctrl_demo = ['age', 'female', 'married', 'size', 'bothwork', 'notemployed']
xs = inc / 100000.0
print('CHJ n', len(y), 'income range', inc.min(), inc.max(), 'share income>200000', np.mean(inc > 200000), 'mean transfers', y.mean())
res = {}
for lab, cols in [('none', []), ('paper', ctrl_full), ('demo', ctrl_demo)]:
    C = d[cols].values.astype(float) if cols else np.zeros((len(y), 0))
    rows = []
    for p in range(1, 13):
        r = base.ols(y, np.column_stack([poly(xs, p), C])); rows.append((p, r['cv'], r['aic']))
        res[(lab, p)] = r
    pcv = min(rows, key=lambda t: t[1])[0]; paic = min(rows, key=lambda t: t[2])[0]
    print('controls', lab, 'CV order', pcv, 'AIC order', paic, [(p, round(cv / 1e8, 5)) for p, cv, _ in rows])
    res[(lab, 'p')] = pcv
    res[(lab, 'C')] = C
# compare control sets at their CV order
for lab in ['none', 'paper', 'demo']:
    p = res[(lab, 'p')]; print(lab, 'best CV', res[(lab, p)]['cv'] / 1e8)
grid = np.linspace(0, 200000, 401); gs = grid / 100000.0
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, lab, t in [(ax[0], 'none', '(a) No controls'), (ax[1], 'paper', '(b) Controls as in the paper')]:
    p = res[(lab, 'p')]; r = res[(lab, p)]; C = res[(lab, 'C')]
    G = np.column_stack([poly(gs, p), np.tile(C.mean(0), (len(gs), 1))]) if C.shape[1] else poly(gs, p)
    m, s = pred(r, G)
    a.fill_between(grid / 1000, m - 1.96 * s, m + 1.96 * s, color='0.82', lw=0); a.plot(grid / 1000, m, 'k', lw=1.4, label='polynomial, order %d' % p)
    # linear spline of Figure 20.2(b)
    kn = [10000, 20000, 50000, 100000, 150000]
    Xs = np.column_stack([np.ones_like(inc), inc] + [(inc - k) * (inc > k) for k in kn] + ([C] if C.shape[1] else []))
    rs = base.ols(y, Xs)
    Gs = np.column_stack([np.ones_like(grid), grid] + [(grid - k) * (grid > k) for k in kn] + ([np.tile(C.mean(0), (len(grid), 1))] if C.shape[1] else []))
    ms, _ = pred(rs, Gs); a.plot(grid / 1000, ms, 'k--', lw=1, label='linear spline (Fig. 20.2b)')
    a.set_title(t + ', order %d' % p, fontsize=10); a.set_xlabel('Total income (thousand pesos)'); a.legend(frameon=False, fontsize=8)
    for q in [0, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 200000]:
        print(' ', lab, q, round(float(np.interp(q, grid, m))), round(float(np.interp(q, grid, s))), 'spline', round(float(np.interp(q, grid, ms))))
    # slope over 0-10000 and 10000-20000
    print(' ', lab, 'avg slope 0-10k %.3f, 10k-20k %.3f, 20k-50k %.3f' % ((np.interp(1e4, grid, m) - np.interp(0, grid, m)) / 1e4, (np.interp(2e4, grid, m) - np.interp(1e4, grid, m)) / 1e4, (np.interp(5e4, grid, m) - np.interp(2e4, grid, m)) / 3e4))
ax[0].set_ylabel('Transfers (pesos)'); plt.tight_layout(); plt.savefig(R + 'ch20_17.pdf'); plt.close()

# ---------------- AL1999
a = pd.read_stata('progs/AL1999.dta')
a['z'] = a.enrollment / np.floor(1 + (a.enrollment - 1) / 40)
a['z1'] = a.z / 40; a['c1'] = a.classize / 40; a['d1'] = a.disadvantaged / 14
a['fourth'] = (a.grade == 4).astype(float)
cl = pd.factorize(a.schlcode)[0]


def tsls(y, Xen, Xex, Zex):
    X = np.column_stack([Xen, Xex]); Z = np.column_stack([Zex, Xex])
    PZX = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(PZX, y, rcond=None)[0]
    e = y - X @ b
    A = np.linalg.inv(PZX.T @ PZX)
    u = pd.DataFrame(PZX * e[:, None]).groupby(cl).sum().values
    G = u.shape[0]; n, k = X.shape
    V = A @ (u.T @ u) @ A * G / (G - 1) * (n - 1) / (n - k)
    return b, V


for dep in ['avgverb', 'avgmath']:
    y = a[dep].values.astype(float)
    Xen = np.column_stack([a.c1, a.c1 ** 2, a.c1 ** 3, a.c1 * a.d1])
    Xex = np.column_stack([a.d1, a.d1 ** 2, a.d1 ** 3, a.enrollment, a.fourth, np.ones(len(a))])
    Zex = np.column_stack([a.z1, a.z1 ** 2, a.z1 ** 3, a.z1 * a.d1])
    b, V = tsls(y, Xen, Xex, Zex)
    se = np.sqrt(np.diag(V))
    names = ['c', 'c2', 'c3', 'cd', 'd', 'd2', 'd3', 'enroll', 'grade4', 'const']
    print(dep, ' '.join('%s=%.3f(%.3f)' % (nm, bb, ss) for nm, bb, ss in zip(names, b, se)))
    idx = [0, 1, 2, 3]; W = b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx])
    from scipy.stats import chi2
    print('  Wald classize', round(W, 2), 'p', round(chi2.sf(W, 4), 4))
    dbar = a.d1.mean()
    g = np.zeros(10); g[0] = 0.5; g[1] = 0.75; g[2] = 0.875; g[3] = 0.5 * dbar
    print('  theta 20->40 (d at mean %.3f): %.3f (%.3f)' % (dbar, g @ b, np.sqrt(g @ V @ g)))
    g1 = g.copy(); g1[3] = 0.5
    print('  theta with d=1 : %.3f (%.3f)' % (g1 @ b, np.sqrt(g1 @ V @ g1)))
    # linear IV (AL Table VII eq 6 style)
    bl, Vl = tsls(y, np.column_stack([a.classize, a.classize * a.disadvantaged]), np.column_stack([a.disadvantaged, a.fourth, a.enrollment, np.ones(len(a))]), np.column_stack([a.z, a.z * a.disadvantaged]))
    print('  linear IV classize %.3f (%.3f), cd %.4f (%.4f)' % (bl[0], np.sqrt(Vl[0, 0]), bl[1], np.sqrt(Vl[1, 1])))
    # effect curves at means
    cg = np.linspace(20, 40, 101) / 40; dg = np.linspace(0, 50, 101) / 14
    xm = dict(d1=a.d1.mean(), c1=a.c1.mean(), enroll=a.enrollment.mean(), fourth=a.fourth.mean())
    Gc = np.column_stack([cg, cg ** 2, cg ** 3, cg * xm['d1'], np.full_like(cg, xm['d1']), np.full_like(cg, xm['d1'] ** 2), np.full_like(cg, xm['d1'] ** 3), np.full_like(cg, xm['enroll']), np.full_like(cg, xm['fourth']), np.ones_like(cg)])
    Gd = np.column_stack([np.full_like(dg, xm['c1']), np.full_like(dg, xm['c1'] ** 2), np.full_like(dg, xm['c1'] ** 3), xm['c1'] * dg, dg, dg ** 2, dg ** 3, np.full_like(dg, xm['enroll']), np.full_like(dg, xm['fourth']), np.ones_like(dg)])
    mc = Gc @ b; sc = np.sqrt(np.einsum('ij,jk,ik->i', Gc, V, Gc)); md = Gd @ b; sd = np.sqrt(np.einsum('ij,jk,ik->i', Gd, V, Gd))
    base.__dict__['AL_' + dep] = (cg * 40, mc, sc, dg * 14, md, sd)
    print('  curve classize 20,25,30,35,40:', np.round(np.interp([20, 25, 30, 35, 40], cg * 40, mc), 2), ' se', np.round(np.interp([20, 30, 40], cg * 40, sc), 2))
    print('  curve disadv 0,10,20,30,40,50:', np.round(np.interp([0, 10, 20, 30, 40, 50], dg * 14, md), 2), ' se', np.round(np.interp([0, 25, 50], dg * 14, sd), 2))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for dep, col, lab in [('avgverb', '0.55', 'reading'), ('avgmath', 'k', 'math')]:
    cg, mc, sc, dg, md, sd = base.__dict__['AL_' + dep]
    ax[0].fill_between(cg, mc - 1.96 * sc, mc + 1.96 * sc, color=col, alpha=0.2, lw=0); ax[0].plot(cg, mc, color=col, lw=1.4, label=lab)
    ax[1].fill_between(dg, md - 1.96 * sd, md + 1.96 * sd, color=col, alpha=0.2, lw=0); ax[1].plot(dg, md, color=col, lw=1.4, label=lab)
ax[0].set_title('(a) Effect of class size', fontsize=10); ax[0].set_xlabel('Class size'); ax[0].set_ylabel('Test score'); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_title('(b) Effect of percent disadvantaged', fontsize=10); ax[1].set_xlabel('Percentage disadvantaged'); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(R + 'ch20_18.pdf'); plt.close()
