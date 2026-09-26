import os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')
LAB = ['married', 'separated', 'divorced', 'never married']


def mnl(y, X, J, base=0):
    """multinomial logit with alternative-specific coefficients; beta_base = 0. returns B (J x k), V (for free params), ll"""
    n, k = X.shape; free = [j for j in range(J) if j != base]
    Y = np.eye(J)[y]
    def unpack(t):
        B = np.zeros((J, k)); B[free] = t.reshape(len(free), k); return B
    def f(t):
        V = X @ unpack(t).T; ls = logsumexp(V, 1)
        P = np.exp(V - ls[:, None])
        g = ((Y - P)[:, free]).T @ X
        return -(np.sum(V * Y) - ls.sum()), -g.ravel()
    r = minimize(f, np.zeros(len(free) * k), jac=True, method='BFGS', options={'gtol': 1e-9, 'maxiter': 10000})
    B = unpack(r.x); V = X @ B.T; P = np.exp(V - logsumexp(V, 1)[:, None]); Pf = P[:, free]
    m = len(free); H = np.zeros((m * k, m * k))
    for a in range(m):
        for b in range(m):
            w = Pf[:, a] * ((a == b) - Pf[:, b])
            H[a * k:(a + 1) * k, b * k:(b + 1) * k] = (X * w[:, None]).T @ X
    return B, np.linalg.inv(H), -r.fun, free


def spline(age):
    a = age / 100
    return np.column_stack([np.ones_like(a), a, a ** 2, ((a - 0.4) ** 2) * (age > 40)])


c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c.columns = ['age', 'female', 'hisp', 'education', 'earnings', 'hours', 'week', 'union', 'uncov', 'region', 'race', 'marital']
# book coding (Figure 26.1): married = marital<=4 (includes widowed), separated=6, divorced=5, never married=7
c['ms'] = np.select([c.marital <= 4, c.marital == 6, c.marital == 5, c.marital == 7], [0, 1, 2, 3])
ag = np.arange(20, 81)
res = {}
for lab, f in [('women', 1), ('men', 0)]:
    s = c[(c.education == 16) & (c.female == f)]
    X = spline(s.age.values.astype(float)); y = s.ms.values
    B, V, ll, free = mnl(y, X, 4, base=0)
    G = spline(ag.astype(float)); Pg = np.exp(G @ B.T - logsumexp(G @ B.T, 1)[:, None])
    res[lab] = Pg
    print('26.12', lab, 'n', len(y), 'shares', np.round(np.bincount(y, minlength=4) / len(y), 3), 'LL %.1f' % ll)
    for a in [25, 30, 40, 50, 60, 70, 80]:
        print('   age', a, np.round(Pg[a - 20], 3))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sty = ['k-', 'k:', 'k--', 'k-.']
for a_, lab in zip(ax, ['men', 'women']):
    for j in range(4):
        a_.plot(ag, res[lab][:, j], sty[j], lw=1.4, label=LAB[j])
    a_.set_title('College-educated %s' % lab, fontsize=10); a_.set_xlabel('Age'); a_.set_ylim(0, 1)
ax[0].set_ylabel('Probability'); ax[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(R + 'ch26_12.pdf'); plt.close()

# 26.13 women <= 35, linear age and education, base = married
s = c[(c.female == 1) & (c.age <= 35)]
X = np.column_stack([np.ones(len(s)), s.age, s.education]).astype(float); y = s.ms.values
B, V, ll, free = mnl(y, X, 4, base=0); se = np.sqrt(np.diag(V)).reshape(3, 3)
print('26.13 n', len(y), 'shares', np.round(np.bincount(y, minlength=4) / len(y), 3), 'LL %.1f' % ll)
for i, j in enumerate(free):
    print('   %s vs married: const %.3f (%.3f) age %.4f (%.4f) edu %.4f (%.4f)' % (LAB[j], B[j, 0], se[i, 0], B[j, 1], se[i, 1], B[j, 2], se[i, 2]))
P = np.exp(X @ B.T - logsumexp(X @ B.T, 1)[:, None])
for k, nm in [(1, 'age'), (2, 'edu')]:
    bbar = (P * B[:, k][None, :]).sum(1)
    ame = (P * (B[:, k][None, :] - bbar[:, None])).mean(0)
    print('   AME', nm, np.round(ame, 4))
# predicted probabilities at age 25, 30, 35 for edu 12 and 16
for a in [25, 30, 35]:
    for e in [12, 16]:
        x = np.array([1, a, e]); p = np.exp(x @ B.T - logsumexp(x @ B.T))
        print('   age %d edu %d:' % (a, e), np.round(p, 3))

# 26.14 nested logit, all women, age spline; groups
s = c[c.female == 1]
X = spline(s.age.values.astype(float)); y = s.ms.values; n, k = X.shape
Bm, Vm, llm, free = mnl(y, X, 4, base=0)
print('26.14 women n', n, 'MNL LL %.2f' % llm)


def nested_ll(t, groups, free_tau):
    B = np.zeros((4, k)); B[1:] = t[:3 * k].reshape(3, k)
    Vv = X @ B.T; taus = []; it = 3 * k
    for fr in free_tau:
        taus.append(t[it] if fr else 1.0); it += fr
    IV = np.column_stack([logsumexp(Vv[:, g] / tau, 1) for g, tau in zip(groups, taus)])
    taus = np.array(taus); den = logsumexp(IV * taus, 1)
    lp = np.zeros((n, 4))
    for gi, (g, tau) in enumerate(zip(groups, taus)):
        for kk in g:
            lp[:, kk] = Vv[:, kk] / tau - IV[:, gi] + tau * IV[:, gi] - den
    return -lp[np.arange(n), y].sum()


for groups, name in [([[0], [3], [1, 2]], '{married},{never},{separated,divorced}'),
                     ([[0, 1], [2, 3]], '{married,separated},{divorced,never}'),
                     ([[0, 1, 2], [3]], '{married,separated,divorced},{never}')]:
    free_tau = [len(g) > 1 for g in groups]
    t0 = np.r_[Bm[1:].ravel(), [0.8] * sum(free_tau)]
    f = lambda t: nested_ll(t, groups, free_tau)
    r = minimize(f, t0, method='BFGS', options={'maxiter': 20000, 'gtol': 1e-7})
    r = minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 40000, 'xatol': 1e-9, 'fatol': 1e-9})
    r = minimize(f, r.x, method='BFGS', options={'maxiter': 20000, 'gtol': 1e-8})
    # SE for tau via numerical hessian
    th = r.x; kk_ = len(th); h = 1e-4; H = np.zeros((kk_, kk_))
    for i in range(kk_):
        for j in range(i, kk_):
            ei = np.zeros(kk_); ej = np.zeros(kk_); ei[i] = h * max(1, abs(th[i])); ej[j] = h * max(1, abs(th[j]))
            H[i, j] = H[j, i] = (f(th + ei + ej) - f(th + ei - ej) - f(th - ei + ej) + f(th - ei - ej)) / (4 * ei[i] * ej[j])
    Vt = np.linalg.pinv(H); taus = th[3 * k:]; set_ = np.sqrt(np.diag(Vt))[3 * k:]
    print('   nested', name, 'LL %.2f' % (-r.fun), 'tau', np.round(taus, 3), 'se', np.round(set_, 3), 'LR vs MNL %.2f' % (2 * (-r.fun - llm)))
