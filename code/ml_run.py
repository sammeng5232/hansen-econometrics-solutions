import numpy as np, sys
from scipy.stats import norm
from choice import *
spec = sys.argv[1]; R = 200
# usage: python ml_run.py base|time|lognormal   (Exercise 26.17 (a), (b), (c))
if spec == 'base':
    X, W, y = load(); ln = False; R = 400
    th0 = np.r_[-0.0218, -0.0149, 0.0356, 0.2946, -2.1494, -0.0507, -0.2343, -1.7901, 0.0081, -0.9887, 1.8621, 0.005]
elif spec == 'time':
    X, W, y = load(altvars=('cost', 'intime', 'outtime')); X = np.stack([X[:, :, 0], X[:, :, 1] + X[:, :, 2]], axis=2); ln = False
    th0 = np.r_[-0.02, -0.012, 0.03, 0.3, -2, -0.05, -0.2, -1.8, 0.008, -1, 1.9, 0.004]
elif spec == 'lognormal':
    X, W, y = load(); X = X.copy(); X[:, :, 1] = -X[:, :, 1]; ln = True
    th0 = np.r_[-0.0226, np.log(0.0168), 0.040, 0.355, -2.72, -0.05, -0.24, -1.82, 0.0084, -1.01, 1.89, 0.3]
N = len(y); draws = norm.ppf(halton_fast(N * R, 3).reshape(N, R))
f = lambda t: nll_mixed(t, X, W, y, 1, draws, lognormal=ln)
th, se, ll = fit_quick(f, th0)
report('MIXED %s R=%d' % (spec, R), th, se, ll, NAMES + ['sigma'])
if ln:
    mu, s = th[1], th[11]; mean = np.exp(mu + s ** 2 / 2); med = np.exp(mu); sd = mean * np.sqrt(np.exp(s ** 2) - 1)
    print('lognormal: mean %.4f median %.4f sd %.4f' % (mean, med, sd))
