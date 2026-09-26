import numpy as np, sys
from choice import *
spec = sys.argv[1]; R = 300
tr = (lambda X: np.log(X)) if spec == 'log' else None
X, W, y = load(transform=tr); N = len(y)
U = np.stack([halton_fast(N * R, b).reshape(N, R) for b in (2, 3, 5)], axis=2)
import os; th_prev = np.load('mnp_%s.npy' % spec) if (spec == 'base' and os.path.exists('mnp_base.npy')) else None
f = lambda t: nll_mnp_general3(t, X, W, y, U)
starts = []
if th_prev is not None:
    S = sigma_chol(th_prev[11:]); M = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1.0]]); Om = M @ S @ M.T
    L = np.linalg.cholesky(Om); scale = np.sqrt(2) / L[0, 0]
    starts.append(np.r_[th_prev[:11] * scale, (L * scale)[1, 0], (L * scale)[1, 1], (L * scale)[2, 0], (L * scale)[2, 1], (L * scale)[2, 2]])
else:
    ts, _, _ = fit_quick(lambda t: nll_mnp_simple(t, X, W, y), np.zeros(11), hess=False)
    starts.append(np.r_[ts, 0.7, 1.2, 0.7, 0.0, 1.2])
starts.append(starts[0] + np.r_[np.zeros(11), 0.3, 0.0, -0.3, 0.2, 0.0])
best = None
for s0 in starts:
    r = minimize(f, s0, method='BFGS', options={'maxiter': 3000, 'gtol': 1e-5})
    print('start LL %.2f' % -r.fun, flush=True)
    if best is None or r.fun < best.fun: best = r
H = num_hess(f, best.x); se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
report('MNP general(diff) %s R=%d' % (spec, R), best.x, se, -best.fun, NAMES + ['L21', 'L22', 'L31', 'L32', 'L33'])
Om = omega_diff(best.x[11:]); print('Omega (air,bus,car minus train)'); print(np.round(Om, 3))
np.save('mnp3_%s.npy' % spec, best.x)
