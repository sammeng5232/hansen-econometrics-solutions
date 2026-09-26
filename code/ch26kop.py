"""Exercises 26.15-26.16 and the simple multinomial probit column of Table 26.1 (Koppelman data).
Conditional logit variants, nested logit variants, the profile likelihood in tau(train,bus), simple MNP."""
import numpy as np
from choice import *

X, W, y = load(altvars=('cost', 'intime', 'outtime'))
X2 = X[:, :, :2]
NM2 = ['g1', 'g2'] + NAMES[2:]

# 26.15 conditional logit
specs = {'(a) cost,intime': X2, '(b) +outtime': X,
         '(c) cost,time=in+out': np.stack([X[:, :, 0], X[:, :, 1] + X[:, :, 2]], 2),
         '(d) log cost, log intime': np.log(X2)}
for lab, Xs in specs.items():
    K = Xs.shape[2]; nm = NM2 if K == 2 else ['cost', 'intime', 'outtime'] + NAMES[2:]
    th, se, ll = fit_quick(lambda t: nll_clogit(t, Xs, W, y), np.zeros(K + 9)); report('CL ' + lab, th, se, ll, nm)

# 26.16 nested logit (Theorem 26.2 form); alternatives 0=train 1=air 2=bus 3=car
for lab, Xs, groups, freet in [('(a) {car,air},{train,bus} tau_tb=1', X2, [[1, 3], [0, 2]], [True, False]),
                               ('(b) logs', np.log(X2), [[1, 3], [0, 2]], [True, False]),
                               ('(c) {car},{train,bus,air}', X2, [[3], [0, 2, 1]], [False, True]),
                               ('(d) {air},{train,bus,car}', X2, [[1], [0, 2, 3]], [False, True])]:
    f = lambda t: nll_nested(t, Xs, W, y, groups, freet)
    best = None
    for tau0 in [0.3, 0.7, 1.2]:
        try:
            th, se, ll = fit_quick(f, np.r_[np.zeros(11), tau0])
            if best is None or ll > best[2]: best = (th, se, ll)
        except Exception as e: print('fail', e)
    report('NL ' + lab, *best, NM2 + ['tau'])

# 26.16(a): why tau(train,bus) must be constrained -- interior local maximum and profile log-likelihood
f = lambda t: nll_nested(t, X2, W, y, [[1, 3], [0, 2]], [True, True])
th, se, ll = fit_quick(f, np.r_[np.zeros(11), 0.3, 0.7]); report('NL both taus free (local max)', th, se, ll, NM2 + ['tau_ca', 'tau_tb'])
th = np.r_[np.zeros(11), 0.3]
for ttb in [0.1, 0.26, 0.5, 1.0, 1.5, 3.0, 10.0, 30.0, 100.0]:
    f = lambda t: nll_nested(np.r_[t, ttb], X2, W, y, [[1, 3], [0, 2]], [True, True])
    th, se, ll = fit_quick(f, th, hess=False)
    print('profile tau_tb %.2f LL %.2f tau_ca %.3f' % (ttb, ll, th[-1]), flush=True)

# simple multinomial probit (iid N(0,1) errors, Gauss-Hermite), levels and logs
for lab, Xs in [('levels', X2), ('logs', np.log(X2))]:
    th, se, ll = fit_quick(lambda t: nll_mnp_simple(t, Xs, W, y), np.zeros(11)); report('MNP simple ' + lab, th, se, ll, NM2)
