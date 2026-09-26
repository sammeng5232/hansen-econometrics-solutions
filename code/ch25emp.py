import os
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')


def probit(y, X, it=100):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        xb = X @ b; q = 2 * y - 1; lam = norm.pdf(q * xb) / norm.cdf(q * xb)
        g = X.T @ (q * lam); H = (X * (lam * (lam + q * xb))[:, None]).T @ X
        step = np.linalg.solve(H, g); b = b + step
        if np.max(np.abs(step)) < 1e-12: break
    xb = X @ b; q = 2 * y - 1; lam = norm.pdf(q * xb) / norm.cdf(q * xb)
    H = (X * (lam * (lam + q * xb))[:, None]).T @ X; s = X * (q * lam)[:, None]
    Hi = np.linalg.inv(H); V = Hi @ (s.T @ s) @ Hi
    ll = np.sum(norm.logcdf(q * xb))
    return b, V, ll


def ape(b, V, X, j, binary=False):
    """average partial effect of regressor j (derivative, or discrete change for binary), delta-method se"""
    def f(bb):
        if binary:
            X1 = X.copy(); X1[:, j] = 1; X0 = X.copy(); X0[:, j] = 0
            return np.mean(norm.cdf(X1 @ bb) - norm.cdf(X0 @ bb))
        return np.mean(norm.pdf(X @ bb)) * bb[j]
    a = f(b); G = np.array([(f(b + 1e-6 * np.eye(len(b))[k]) - a) / 1e-6 for k in range(len(b))])
    return a, np.sqrt(G @ V @ G)


c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c.columns = ['age', 'female', 'hisp', 'education', 'earnings', 'hours', 'week', 'union', 'uncov', 'region', 'race', 'marital']
c['black'] = (c.race == 2).astype(float)
print('marital codes', c.marital.value_counts().sort_index().to_dict())
# 25.15-25.16 union
for lab, f in [('men', 0), ('women', 1)]:
    s = c[c.female == f]
    y = s.union.values.astype(float)
    X = np.column_stack([np.ones(len(s)), s.age, s.education, s.black, s.hisp]).astype(float)
    b, V, ll = probit(y, X); se = np.sqrt(np.diag(V))
    print('UNION', lab, 'n', len(y), 'mean', y.mean().round(4), ' '.join('%s=%.4f(%.4f)' % (nm, bb, ss) for nm, bb, ss in zip(['c', 'age', 'edu', 'black', 'hisp'], b, se)))
    for j, nm, bi in [(1, 'age', False), (2, 'edu', False), (3, 'black', True), (4, 'hisp', True)]:
        a, sa = ape(b, V, X, j, bi); print('   APE %s %.4f (%.4f)' % (nm, a, sa))
# 25.17 college women marriage vs age
def spl(age):
    a1 = age / 100
    return np.column_stack([np.ones_like(a1), a1, a1 ** 2, ((a1 - 0.4) ** 2) * (age > 40), ((a1 - 0.6) ** 2) * (age > 60)])
xg = np.arange(19, 81)
curves = {}
for lab, f, defn in [('women', 1, 'm123'), ('men', 0, 'm123'), ('men', 0, 'm1234')]:
    s = c[(c.education == 16) & (c.female == f)]
    y = (s.marital <= (3 if defn == 'm123' else 4)).values.astype(float)
    X = spl(s.age.values.astype(float)); b, V, ll = probit(y, X)
    G = spl(xg.astype(float)); p = norm.cdf(G @ b)
    J = norm.pdf(G @ b)[:, None] * G; sp_ = np.sqrt(np.einsum('ij,jk,ik->i', J, V, J))
    # compare with quadratic probit and linear-in-age by AIC
    Xq = X[:, :3]; bq, Vq, llq = probit(y, Xq)
    curves[(lab, defn)] = (p, sp_)
    raw = s.groupby('age').apply(lambda g: (g.marital <= (3 if defn == 'm123' else 4)).mean())
    print('MARRIAGE college', lab, defn, 'n', len(y), 'mean', y.mean().round(3), 'loglik spline %.1f quad %.1f  AIC spline %.1f quad %.1f' % (ll, llq, -2 * ll + 10, -2 * llq + 6))
    print('   P at ages', {a: round(float(p[a - 19]), 3) for a in [22, 25, 30, 35, 40, 50, 60, 70, 80]})
    curves[(lab, defn, 'raw')] = raw
fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
p, s_ = curves[('women', 'm123')]
ax.fill_between(xg, p - 1.96 * s_, p + 1.96 * s_, color='0.82', lw=0); ax.plot(xg, p, 'k', lw=1.5, label='women (marital 1-3)')
p2, _ = curves[('men', 'm123')]; ax.plot(xg, p2, 'k--', lw=1.2, label='men (marital 1-3)')
p3, _ = curves[('men', 'm1234')]; ax.plot(xg, p3, 'k:', lw=1.2, label='men, Figure 25.1 definition (1-4)')
raw = curves[('women', 'm123', 'raw')]; ax.plot(raw.index, raw.values, '.', color='0.5', ms=4, label='women, raw proportion by age')
ax.set_xlabel('Age'); ax.set_ylabel('Probability married'); ax.set_ylim(0, 1); ax.set_xlim(19, 80); ax.legend(frameon=False, fontsize=8, loc='lower center')
plt.tight_layout(); plt.savefig(R + 'ch25_17.pdf'); plt.close()
# 25.18-25.19 marriage with age spline + education + black + hisp
for lab, f in [('men', 0), ('women', 1)]:
    s = c[c.female == f]
    y = (s.marital <= 3).values.astype(float)
    A = spl(s.age.values.astype(float))
    X = np.column_stack([A, s.education, s.black, s.hisp]).astype(float)
    b, V, ll = probit(y, X); se = np.sqrt(np.diag(V))
    print('MARRIED', lab, 'n', len(y), 'mean', y.mean().round(3), ' '.join('%s=%.4f(%.4f)' % (nm, bb, ss) for nm, bb, ss in zip(['c', 'a', 'a2', 's40', 's60', 'edu', 'black', 'hisp'], b, se)))
    for j, nm, bi in [(5, 'edu', False), (6, 'black', True), (7, 'hisp', True)]:
        a, sa = ape(b, V, X, j, bi); print('   APE %s %.4f (%.4f)' % (nm, a, sa))
    # predicted profile at mean edu, non-black non-hisp
    G = np.column_stack([spl(xg.astype(float)), np.full(len(xg), s.education.mean()), np.zeros(len(xg)), np.zeros(len(xg))])
    p = norm.cdf(G @ b); print('   P (white non-Hisp, mean edu) at ages', {a: round(float(p[a - 19]), 3) for a in [25, 30, 40, 50, 60, 70]})
