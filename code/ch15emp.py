import os
import numpy as np, pandas as pd, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from varlib import *
from scipy import stats

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'figure.dpi': 100})
q = pd.read_stata('progs/FRED-QD.dta'); m = pd.read_stata('progs/FRED-MD.dta'); kil = pd.read_stata('progs/Kilian2009.dta')
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500

def panel(fname, R, lo, hi, pairs, titles, xlabel, ncol=3):
    n = len(pairs); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.8 * nrow), squeeze=False)
    H = R.shape[0] - 1; h = np.arange(H + 1)
    for k, ((i, j), t) in enumerate(zip(pairs, titles)):
        ax = axes[k // ncol][k % ncol]
        ax.fill_between(h, lo[:, i, j], hi[:, i, j], color='0.85', lw=0)
        ax.plot(h, R[:, i, j], 'k-', lw=1.2); ax.axhline(0, color='k', ls=':', lw=0.7)
        ax.set_title(t); ax.set_xlabel(xlabel)
    for k in range(n, nrow * ncol): axes[k // ncol][k % ncol].axis('off')
    fig.tight_layout(); fig.savefig(FIG + fname); plt.close(fig)

def show(label, R, pairs, names, hs):
    print('  ', label)
    for (i, j), nm in zip(pairs, names):
        print(f'     {nm:32s}', ' '.join(f'h{h}:{R[h,i,j]: .3f}' for h in hs))

# ---------------- 15.14 ----------------
print('=== 15.14')
g = 100*np.log(q.gdpc1.values.astype(float)); pr = 100*np.log(q.gdpctpi.values.astype(float))
Y = np.column_stack([np.diff(g), np.diff(pr), q.fedfunds.values.astype(float)[1:]])
v = var_est(Y, 6); Bc = np.linalg.cholesky(v['Sigma']); H = 20
R = irf(v['A'], Bc, H, cumulative=[0, 1])
lo, hi = bootstrap_bands(Y, 6, lambda vv: np.linalg.cholesky(vv['Sigma']), H, REPS, 1514, cumulative=[0, 1])
pairs = [(0, 0), (1, 0), (2, 0)]; names = ['GDP level (cum.)', 'price level (cum.)', 'fed funds']
show('supply shock (1 s.d. GDP-equation shock)', R, pairs, names, [0, 1, 2, 4, 8, 12, 20])
print('     impact sd of shock', np.sqrt(v['Sigma'][0, 0]), ' n', v['n'])
panel('ch15_14.pdf', R, lo, hi, pairs, ['GDP (level) to supply shock', 'Price level to supply shock', 'Fed funds to supply shock'], 'quarters')
for pr_ in pairs: print('     90% band at h=8', lo[8][pr_], hi[8][pr_], ' h=20', lo[20][pr_], hi[20][pr_])

# ---------------- 15.15 ----------------
print('=== 15.15')
YK = np.column_stack([-kil.oil.values.astype(float), kil.output.values.astype(float), kil.price.values.astype(float)])
v = var_est(YK, 4); Bc = np.linalg.cholesky(v['Sigma']); H = 24
R = irf(v['A'], Bc, H)
lo, hi = bootstrap_bands(YK, 4, lambda vv: np.linalg.cholesky(vv['Sigma']), H, REPS, 1515)
pairs = [(1, 0), (1, 1), (1, 2)]; names = ['output <- oil supply shock', 'output <- aggregate demand shock', 'output <- oil-specific demand']
show('orthogonalized', R, pairs, names, [0, 1, 3, 6, 12, 18, 24])
for pr_ in pairs: print('     90% band h=6', lo[6][pr_], hi[6][pr_], ' h=12', lo[12][pr_], hi[12][pr_], ' h=24', lo[24][pr_], hi[24][pr_])
print('     price responses', np.round(R[[0, 3, 6, 12, 24], 2, 0], 2), np.round(R[[0, 3, 6, 12, 24], 2, 1], 2), np.round(R[[0, 3, 6, 12, 24], 2, 2], 2))
panel('ch15_15.pdf', R, lo, hi, pairs, ['Output to oil supply shock', 'Output to aggregate demand shock', 'Output to oil-specific demand shock'], 'months')

# ---------------- 15.16 ----------------
print('=== 15.16')
mm = m[['time', 'permit', 'houst', 'realln']].copy()
rl = 100*np.diff(np.log(mm.realln.values.astype(float)))
Y = np.column_stack([mm.permit.values.astype(float)[1:], mm.houst.values.astype(float)[1:], rl])
ok = ~np.isnan(Y).any(1); first = np.argmax(ok); Y = Y[first:]
print('   sample start', mm.time.values[first + 1], ' T', len(Y))
for p, n, a in aic_var(Y, 8): print(f'     VAR({p}) n={n} AIC={a:.2f}')
aics = aic_var(Y, 8); psel = min(aics, key=lambda z: z[2])[0]; print('   selected p', psel)
v = var_est(Y, psel); Bc = np.linalg.cholesky(v['Sigma']); H = 36
R = irf(v['A'], Bc, H)
lo, hi = bootstrap_bands(Y, psel, lambda vv: np.linalg.cholesky(vv['Sigma']), H, REPS, 1516)
pairs = [(1, 0), (1, 1), (1, 2)]; names = ['houst <- permit shock', 'houst <- houst shock', 'houst <- loan shock']
show('orthogonalized', R, pairs, names, [0, 1, 3, 6, 12, 24, 36])
for pr_ in pairs: print('     90% band h=6', lo[6][pr_], hi[6][pr_], ' h=24', lo[24][pr_], hi[24][pr_])
print('   sd of shocks', np.sqrt(np.diag(v['Sigma'])), ' corr', np.round(v['Sigma'][0, 1]/np.sqrt(v['Sigma'][0, 0]*v['Sigma'][1, 1]), 3))
panel('ch15_16.pdf', R, lo, hi, pairs, ['Housing starts to permit shock', 'Housing starts to own shock', 'Housing starts to loan-growth shock'], 'months')

# ---------------- 15.17 ----------------
print('=== 15.17')
inv = 100*np.log(q.gpdic1.values.astype(float)); P = 100*np.log(q.gdpctpi.values.astype(float)); G = 100*np.log(q.gdpc1.values.astype(float)); FF = q.fedfunds.values.astype(float)
Y = np.column_stack([inv, P, G, FF])
print('   AIC (with trend):', [(p, round(a, 1)) for p, n, a in aic_var(Y, 8, trend=True)])
v = var_est(Y, 6, trend=True)
nan = np.nan
T17 = np.array([[1, 0, -1, 0], [0, 1, nan, 0], [nan, nan, 1, 0], [nan, nan, nan, 1]], dtype=float)
A, D, cost = svar_A(v['Sigma'], T17)
print('   A =\n', np.round(A, 4), '\n   diag var eps', np.round(np.diag(D), 4), ' cost', cost)
Bimp = np.linalg.inv(A) @ np.sqrt(D)
H = 20; R = irf(v['A'], Bimp, H)
def B17(vv):
    Aa, Dd, _ = svar_A(vv['Sigma'], T17); return np.linalg.inv(Aa) @ np.sqrt(Dd)
lo, hi = bootstrap_bands(Y, 6, B17, H, max(100, REPS // 5), 1517, trend=True)
pairs = [(2, 3), (2, 2), (1, 2)]; names = ['GDP <- fed funds shock', 'GDP <- GDP shock', 'price <- GDP shock']
show('structural', R, pairs, names, [0, 1, 2, 4, 8, 12, 20])
for pr_ in pairs: print('     90% band h=4', lo[4][pr_], hi[4][pr_], ' h=12', lo[12][pr_], hi[12][pr_])
panel('ch15_17.pdf', R, lo, hi, pairs, ['GDP to fed funds shock', 'GDP to GDP shock', 'Price level to GDP shock'], 'quarters')

# ---------------- 15.18 ----------------
print('=== 15.18')
v = var_est(YK, 4)
T18 = np.array([[1, 0, 0], [0, 1, nan], [nan, nan, 1]], dtype=float)
A, D, cost = svar_A(v['Sigma'], T18)
print('   A =\n', np.round(A, 4), '\n   diag var eps', np.round(np.diag(D), 4), ' cost', cost)
Bimp = np.linalg.inv(A) @ np.sqrt(D); H = 24; R = irf(v['A'], Bimp, H)
def B18(vv):
    Aa, Dd, _ = svar_A(vv['Sigma'], T18); return np.linalg.inv(Aa) @ np.sqrt(Dd)
lo, hi = bootstrap_bands(YK, 4, B18, H, max(100, REPS // 5), 1518)
pairs = [(2, 0), (2, 1), (2, 2)]; names = ['price <- supply', 'price <- agg demand', 'price <- oil demand']
show('structural', R, pairs, names, [0, 1, 3, 6, 12, 18, 24])
for pr_ in pairs: print('     90% band h=6', lo[6][pr_], hi[6][pr_], ' h=24', lo[24][pr_], hi[24][pr_])
Rc = irf(v['A'], np.linalg.cholesky(v['Sigma']), H); print('   recursive (15.15) price responses h=0,6,24', np.round(Rc[[0, 6, 24], 2, :], 2))
panel('ch15_18.pdf', R, lo, hi, pairs, ['Oil price to supply shock', 'Oil price to aggregate demand shock', 'Oil price to oil-specific demand shock'], 'months')

# ---------------- 15.19 ----------------
print('=== 15.19')
gg = 100*np.diff(np.log(q.gdpc1.values.astype(float)))
M1 = q.m1realx.values.astype(float)*q.cpiaucsl.values.astype(float)
mg = 100*np.diff(np.log(M1))
n = len(gg)
X = np.column_stack([np.ones(n - 4)] + [gg[4 - l:n - l] for l in range(1, 5)] + [mg[4 - l:n - l] for l in range(1, 5)])
y = gg[4:]
XXi = np.linalg.inv(X.T @ X); b = XXi @ (X.T @ y); e = y - X @ b; k = X.shape[1]
Vr = (len(y)/(len(y) - k))*XXi @ ((X*(e**2)[:, None]).T @ X) @ XXi
idx = [5, 6, 7, 8]; d = b[idx]; W = d @ np.linalg.solve(Vr[np.ix_(idx, idx)], d)
print('   n', len(y), ' money coefs', np.round(d, 4), ' se', np.round(np.sqrt(np.diag(Vr)[idx]), 4), ' Wald', W, ' p', stats.chi2.sf(W, 4))
Rv = np.zeros(k); Rv[idx] = 1; s = Rv @ b; se = np.sqrt(Rv @ Vr @ Rv)
print('   sum', s, ' se', se, ' t', s/se, ' p', 2*stats.norm.sf(abs(s/se)), ' gdp lag coefs', np.round(b[1:5], 3))
Y = np.column_stack([gg, mg]); v = var_est(Y, 4)
A1 = np.eye(2) - sum(v['A']); A1i = np.linalg.inv(A1)
C = np.linalg.cholesky(A1i @ v['Sigma'] @ A1i.T); Bimp = A1 @ C
print('   C =', np.round(C, 4), ' B =', np.round(Bimp, 4))
H = 24; R = irf(v['A'], Bimp, H, cumulative=[0, 1])
def B19(vv):
    a1 = np.eye(2) - sum(vv['A']); ai = np.linalg.inv(a1); return a1 @ np.linalg.cholesky(ai @ vv['Sigma'] @ ai.T)
lo, hi = bootstrap_bands(Y, 4, B19, H, REPS, 1519, cumulative=[0, 1])
pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]; names = ['GDP <- shock1 (real)', 'GDP <- shock2 (money)', 'M1 <- shock1', 'M1 <- shock2']
show('long-run identified, levels (cum.)', R, pairs, names, [0, 1, 2, 4, 8, 12, 24])
for pr_ in pairs: print('     90% band h=4', lo[4][pr_], hi[4][pr_], ' h=24', lo[24][pr_], hi[24][pr_])
panel('ch15_19.pdf', R, lo, hi, pairs, ['GDP (level) to real shock', 'GDP (level) to nominal shock', 'Nominal M1 (level) to real shock', 'Nominal M1 (level) to nominal shock'], 'quarters', ncol=2)

# ---------------- 15.20 ----------------
print('=== 15.20')
hrs = 100*np.diff(np.log(q.hoanbs.values.astype(float))); gd = 100*np.diff(np.log(q.gdpc1.values.astype(float)))
dp = 100*np.diff(np.log(q.gdpctpi.values.astype(float))); ddp = np.diff(dp)
Y = np.column_stack([hrs[1:], gd[1:], ddp])
aics = aic_var(Y, 8); print('   AIC', [(p, round(a, 1)) for p, n, a in aics]); psel = min(aics, key=lambda z: z[2])[0]; print('   selected', psel)
v = var_est(Y, psel)
A1 = np.eye(3) - sum(v['A']); A1i = np.linalg.inv(A1)
C = np.linalg.cholesky(A1i @ v['Sigma'] @ A1i.T); Bimp = A1 @ C
print('   C =\n', np.round(C, 4))
H = 24; R = irf(v['A'], Bimp, H, cumulative=[0, 1])
def B20(vv):
    a1 = np.eye(3) - sum(vv['A']); ai = np.linalg.inv(a1); return a1 @ np.linalg.cholesky(ai @ vv['Sigma'] @ ai.T)
lo, hi = bootstrap_bands(Y, psel, B20, H, REPS, 1520, cumulative=[0, 1])
pairs = [(1, 0), (1, 1), (1, 2)]; names = ['GDP <- labor supply', 'GDP <- technology', 'GDP <- demand']
show('long-run identified GDP level', R, pairs, names, [0, 1, 2, 4, 8, 12, 24])
for pr_ in pairs: print('     90% band h=4', lo[4][pr_], hi[4][pr_], ' h=24', lo[24][pr_], hi[24][pr_])
panel('ch15_20.pdf', R, lo, hi, pairs, ['GDP (level) to labor-supply shock', 'GDP (level) to technology shock', 'GDP (level) to demand shock'], 'quarters')
print('done')
