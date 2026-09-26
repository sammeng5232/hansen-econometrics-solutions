import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')
d = pd.read_stata('progs/LM2007.dta')
c = 59.1984
s6 = np.sqrt(6)
tri = lambda u: (1 - np.abs(u) / s6) * (np.abs(u) < s6) / s6


def ll(y, x, g, h):
    out = np.zeros(len(g))
    for j, x0 in enumerate(g):
        u = x - x0; K = tri(u / h); Z = np.column_stack([np.ones_like(u), u])
        out[j] = np.linalg.solve((Z * K[:, None]).T @ Z, (Z * K[:, None]).T @ y)[0]
    return out


def loo(y, x, h):
    e = np.zeros(len(y))
    for j in range(len(y)):
        u = x - x[j]; K = tri(u / h); K[j] = 0; Z = np.column_stack([np.ones_like(u), u])
        A = (Z * K[:, None]).T @ Z
        if abs(np.linalg.det(A)) < 1e-12:
            e[j] = np.nan; continue
        e[j] = y[j] - np.linalg.solve(A, (Z * K[:, None]).T @ y)[0]
    return e


def llse(y, x, g, h, e):
    s = np.zeros(len(g)); ok = ~np.isnan(e)
    for j, x0 in enumerate(g):
        u = x[ok] - x0; K = tri(u / h); Z = np.column_stack([np.ones_like(u), u])
        A = np.linalg.inv((Z * K[:, None]).T @ Z); ze = Z * (K * e[ok])[:, None]
        s[j] = np.sqrt((A @ ze.T @ ze @ A)[0, 0])
    return s


def rdd(yname, h, grid=False):
    y = d[yname].values.astype(float); x = d.povrate60.values.astype(float); T = x >= c
    y0, x0, y1, x1 = y[~T], x[~T], y[T], x[T]
    m0 = ll(y0, x0, [c], h)[0]; m1 = ll(y1, x1, [c], h)[0]
    e0 = loo(y0, x0, h); e1 = loo(y1, x1, h)
    s0 = llse(y0, x0, [c], h, e0)[0]; s1 = llse(y1, x1, [c], h, e1)[0]
    th = m1 - m0; se = np.sqrt(s0 ** 2 + s1 ** 2)
    out = dict(theta=th, se=se, p=2 * norm.sf(abs(th / se)), m0=m0, m1=m1, n0=int((np.abs(x0 - c) < s6 * h).sum()), n1=int((np.abs(x1 - c) < s6 * h).sum()))
    if grid:
        g0 = np.arange(15, c, 0.4); g1 = np.arange(c, 82, 0.4)
        out.update(g0=g0, g1=g1, f0=ll(y0, x0, g0, h), f1=ll(y1, x1, g1, h), b0=llse(y0, x0, g0, h, e0), b1=llse(y1, x1, g1, h, e1))
    return out


def simple(yname, h):
    y = d[yname].values.astype(float); x = d.povrate60.values.astype(float)
    w = np.abs(x - c) <= h; y = y[w]; x = x[w]; T = (x >= c).astype(float)
    Z = np.column_stack([np.ones_like(x), x, (x - c) * T, T])
    ZZ = np.linalg.inv(Z.T @ Z); b = ZZ @ Z.T @ y; e = y - Z @ b
    V = ZZ @ ((Z * e[:, None]).T @ (Z * e[:, None])) @ ZZ
    return b, np.sqrt(np.diag(V)), len(y)


print('n', len(d), 'treated', (d.povrate60 >= c).sum())
# 21.5
for h in [13.8, 7, 20]:
    b, s, n = simple('mort_age59_related_postHS', h)
    print('21.5 h=%.1f n=%d' % (h, n), ' '.join('%.3f(%.3f)' % (bb, ss) for bb, ss in zip(b, s)), 'p %.3f' % (2 * norm.sf(abs(b[3] / s[3]))))
# 21.6-21.9
outs = {}
for yname in ['mort_age59_related_postHS', 'mort_age59_injury_postHS', 'mort_age25plus_related_postHS', 'mort_age59_related_preHS']:
    for h in [4, 8, 12]:
        r = rdd(yname, h, grid=(h == 8)); outs[(yname, h)] = r
        print(yname, 'h=%d theta %.3f (%.3f) p %.3f  m0(c)=%.2f  n_eff %d/%d' % (h, r['theta'], r['se'], r['p'], r['m0'], r['n0'], r['n1']), flush=True)
    b, s, n = simple(yname, 13.8)
    print('   simple h=13.8: theta %.3f (%.3f), n=%d' % (b[3], s[3], n))
    y = d[yname]; print('   mean', y.mean(), 'sd', y.std(), 'mean near cutoff (50-59.2)', y[(d.povrate60 > 50) & (d.povrate60 < c)].mean())
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
titles = {'mort_age59_related_postHS': '(a) Ages 5-9, HS-related, 1973-83', 'mort_age59_injury_postHS': '(b) Ages 5-9, injuries, 1973-83',
          'mort_age25plus_related_postHS': '(c) Ages 25+, HS-related, 1973-83', 'mort_age59_related_preHS': '(d) Ages 5-9, HS-related, 1959-64'}
for a, yname in zip(ax.flat, titles):
    r = outs[(yname, 8)]
    for g, f, b in [(r['g0'], r['f0'], r['b0']), (r['g1'], r['f1'], r['b1'])]:
        a.fill_between(g, f - 1.96 * b, f + 1.96 * b, color='0.82', lw=0); a.plot(g, f, 'k', lw=1.4)
    a.axvline(c, color='k', ls=':', lw=0.8); a.set_title(titles[yname] + r'  ($\hat\theta$ = %.2f, s.e. %.2f)' % (r['theta'], r['se']), fontsize=9)
    a.set_xlabel('1960 poverty rate'); a.set_ylabel('Mortality per 100,000'); a.set_xlim(15, 82)
plt.tight_layout(); plt.savefig(R + 'ch21_rdd.pdf'); plt.close()
