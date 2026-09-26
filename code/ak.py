import numpy as np, pandas as pd, sys
from scipy import stats
from ivlib import tsls, liml
from boot import bc

d = pd.read_stata('AK1991/AK1991.dta')
d = d.astype({c: float for c in d.columns})

def dummies(v, levels, drop_first=True):
    lv = levels[1:] if drop_first else levels
    return np.column_stack([(v == l).astype(float) for l in lv])

def indep_cols(M, base_rank_cols=0, tol=1e-8):
    """Greedy selection of linearly independent columns (keeps first base_rank_cols)."""
    q, r = np.linalg.qr(M, mode='reduced')
    diag = np.abs(np.diag(r))
    keep = diag > tol * diag.max()
    return keep

def build(df, black_reg=True, state=False, inst='qobyob'):
    n = len(df); one = np.ones((n, 1))
    yobs = sorted(df.yob.unique()); regs = sorted(df.region.unique()); sts = sorted(df.state.unique())
    ex = [df.smsa.values[:, None], df.married.values[:, None], dummies(df.yob.values, yobs), dummies(df.region.values, regs)]
    if black_reg: ex.insert(0, df.black.values[:, None])
    if state: ex.append(dummies(df.state.values, sts))
    ex.append(one)
    Z1 = np.column_stack(ex)
    Q = dummies(df.qob.values, [1, 2, 3, 4])  # Q2,Q3,Q4
    if inst == 'qob':
        Z2 = Q
    else:
        Yd = dummies(df.yob.values, yobs, drop_first=False)
        parts = [np.column_stack([Q[:, j] * Yd[:, t] for j in range(3) for t in range(Yd.shape[1])])]
        if inst == 'qobyobstate':
            Sd = dummies(df.state.values, sts)
            parts.append(np.column_stack([Q[:, j] * Sd[:, t] for j in range(3) for t in range(Sd.shape[1])]))
        Z2 = np.column_stack(parts)
    return Z1, Z2

def reduce_rank(Z1, Z2):
    Z = np.column_stack([Z1, Z2])
    keep = indep_cols(Z)
    k1 = Z1.shape[1]
    assert keep[:k1].all(), 'exogenous regressors collinear'
    return Z2[:, keep[k1:]]

def rf_F(x, Z1, Z2):
    n = len(x)
    def ssr(A):
        b = np.linalg.lstsq(A, x, rcond=None)[0]; e = x - A @ b; return e @ e
    Z = np.column_stack([Z1, Z2]); q = Z2.shape[1]; k = Z.shape[1]
    sr, su = ssr(Z1), ssr(Z)
    F = ((sr - su) / q) / (su / (n - k))
    return F, q, stats.f.sf(F, q, n - k)

def run(df, label, black_reg, state, inst, do_liml=False, boot=0):
    Z1, Z2 = build(df, black_reg, state, inst)
    Z2 = reduce_rank(Z1, Z2)
    x = df.edu.values; y = df.logwage.values
    F, q, p = rf_F(x, Z1, Z2)
    X = np.column_stack([x, Z1]); Z = np.column_stack([Z1, Z2])
    b, V, e = tsls(y, X, Z, robust=False)
    br, Vr, _ = tsls(y, X, Z, robust=True)
    names = ['edu'] + (['black'] if black_reg else []) + ['smsa', 'married']
    print(f'--- {label}: n={len(df)} excluded instruments q={q}  first-stage F={F:.3f} (p={p:.2g})')
    for i, nm in enumerate(names):
        print(f'    2SLS {nm:8s} {b[i]: .4f}  homo se {np.sqrt(V[i,i]):.4f}  robust se {np.sqrt(Vr[i,i]):.4f}')
    if do_liml:
        bl, Vl, _, kap = liml(y, x[:, None], Z1, Z, robust=False)
        print(f'    LIML edu {bl[0]: .4f} homo se {np.sqrt(Vl[0,0]):.4f} kappa {kap:.5f}')
    if inst == 'qob':
        g = np.linalg.lstsq(Z, x, rcond=None)[0]
        print('    reduced-form Q2,Q3,Q4 coefs:', np.round(g[-3:], 4))
    if boot:
        rng = np.random.default_rng(1227); n = len(df); bs = np.empty(boot)
        for k in range(boot):
            idx = rng.integers(0, n, n)
            bs[k] = tsls(y[idx], X[idx], Z[idx], robust=False)[0][0]
        ci, z0 = bc(bs, b[0])
        print(f'    bootstrap B={boot}: se {bs.std(ddof=1):.4f}  IQR/1.349 {(np.quantile(bs,.75)-np.quantile(bs,.25))/1.349:.4f}  BC [{ci[0]:.4f},{ci[1]:.4f}] z0 {z0:.3f}  pct {np.round(np.quantile(bs,[.025,.975]),4)}  min {bs.min():.3f} max {bs.max():.3f}')

which = sys.argv[1]
if which == 'full':
    run(d, 'FULL (12.91)-(12.92) check', True, False, 'qob')
    run(d, 'FULL (12.89) check', True, False, 'qobyob')
else:
    bm = d[d.black == 1].reset_index(drop=True)
    run(bm, 'Black men, analog of (12.90): 180 instruments + state dummies', False, True, 'qobyobstate', do_liml=True)
    run(bm, 'Black men, analog of (12.89): 30 instruments', False, False, 'qobyob', do_liml=True)
    run(bm, 'Black men, analog of (12.92): 3 instruments', False, False, 'qob', do_liml=True, boot=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
