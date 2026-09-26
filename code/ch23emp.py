import os
import numpy as np, pandas as pd, warnings
from scipy.optimize import least_squares, minimize_scalar
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')


def jac(f, th, eps=1e-6):
    f0 = f(th); J = np.zeros((len(f0), len(th)))
    for j in range(len(th)):
        t = th.copy(); h = eps * max(1, abs(th[j])); t[j] += h; J[:, j] = (f(t) - f0) / h
    return J


def nlls_se(mfun, th, y, cl=None, hc='HC0'):
    e = y - mfun(th); M = jac(mfun, th); n, k = M.shape
    Q = M.T @ M
    if cl is None:
        Om = (M * e[:, None]).T @ (M * e[:, None])
    else:
        u = pd.DataFrame(M * e[:, None]).groupby(cl).sum().values; Om = u.T @ u
    Qi = np.linalg.inv(Q); V = Qi @ Om @ Qi
    return V, e


# ---------------- CES
d = pd.read_stata('progs/PSS2017.dta')
for lab, x1n, x2n in [('capacity', 'EC_c', 'EC_d'), ('alt capital', 'EC_c_alt', 'EC_d_alt')]:
    s = d.dropna(subset=['EG_total', x1n, x2n]).copy()
    sc = 1e9 if 'alt' in x1n else 1.0
    y = np.log(s.EG_total.values); x1 = s[x1n].values / sc; x2 = s[x2n].values / sc; cl = s.country.values
    def m(th):
        rho, nu, al, b = th
        if abs(rho) < 1e-8:
            return b + nu * (al * np.log(x1) + (1 - al) * np.log(x2))
        return b + nu / rho * np.log(al * x1 ** rho + (1 - al) * x2 ** rho)
    best = None
    for r0 in [-2, -1, -0.5, 0.1, 0.3, 0.6, 0.9]:
        for a0 in [0.2, 0.4, 0.6, 0.8]:
            try:
                sol = least_squares(lambda th: y - m(th), [r0, 1.0, a0, np.mean(y)], bounds=([-20, 0, 1e-6, -np.inf], [1.0, 5, 1 - 1e-6, np.inf]), xtol=1e-14, ftol=1e-14)
                if best is None or sol.cost < best.cost: best = sol
            except Exception:
                pass
    th = best.x; V, e = nlls_se(m, th, y, cl)
    se = np.sqrt(np.diag(V)); sig = 1 / (1 - th[0]); sesig = se[0] / (1 - th[0]) ** 2
    # beta at original scale: X/sc => beta_orig = beta - nu*log(sc)
    print('CES', lab, 'n', len(y), 'countries', len(np.unique(cl)), 'rho %.3f (%.3f) nu %.3f (%.3f) alpha %.3f (%.3f) beta %.3f (%.3f) [orig-scale beta %.3f] sigma %.3f (%.3f) SSR %.4f' % (th[0], se[0], th[1], se[1], th[2], se[2], th[3], se[3], th[3] - th[1] * np.log(sc), sig, sesig, 2 * best.cost))
    # also rho profile: min over other params for grid of rho
    prof = []
    for r in np.linspace(-3, 0.95, 80):
        f = lambda t: y - m(np.r_[r, t])
        so = least_squares(f, th[1:], bounds=([0, 1e-6, -np.inf], [5, 1 - 1e-6, np.inf]))
        prof.append((r, 2 * so.cost / len(y)))
    prof = np.array(prof); print('  rho profile min at', prof[prof[:, 1].argmin()])
    # Cobb-Douglas test: rho = 0 t
    print('  t(rho=0) %.2f' % (th[0] / se[0]))

# ---------------- RR kink
rr = pd.read_stata('RR2010/RR2010.dta').sort_values('year')
for lab, yv in [('growth', 'gdp'), ('inflation', 'inflation')]:
    Y = rr[yv].values.astype(float); X = rr.debt.values.astype(float)
    y = Y[1:]; y1 = Y[:-1]; x = X[:-1]
    def design(c):
        return np.column_stack([np.minimum(x - c, 0), np.maximum(x - c, 0), y1, np.ones_like(y)])
    cs = np.linspace(np.quantile(x, 0.05), np.quantile(x, 0.95), 2000)
    ssr = []
    for c in cs:
        Z = design(c); b = np.linalg.lstsq(Z, y, rcond=None)[0]; ssr.append(np.sum((y - Z @ b) ** 2))
    ssr = np.array(ssr); c0 = cs[ssr.argmin()]
    res = minimize_scalar(lambda c: np.sum((y - design(c) @ np.linalg.lstsq(design(c), y, rcond=None)[0]) ** 2), bounds=(c0 - 1, c0 + 1), method='bounded')
    c = res.x; Z = design(c); b = np.linalg.lstsq(Z, y, rcond=None)[0]
    th = np.r_[b, c]
    mf = lambda t: np.column_stack([np.minimum(x - t[4], 0), np.maximum(x - t[4], 0), y1, np.ones_like(y)]) @ t[:4]
    # analytic derivative for c
    e = y - mf(th)
    M = np.column_stack([Z, -th[0] * (x < c) - th[1] * (x > c)])
    Qi = np.linalg.inv(M.T @ M); V = Qi @ ((M * e[:, None]).T @ (M * e[:, None])) @ Qi; se = np.sqrt(np.diag(V))
    print('KINK', lab, 'n', len(y), ' '.join('%s=%.4f(%.4f)' % (nm, t, s) for nm, t, s in zip(['b1', 'b2', 'b3', 'b4', 'c'], th, se)), 'SSR', np.sum(e ** 2))
    # linear model comparison
    Zl = np.column_stack([x, y1, np.ones_like(y)]); bl = np.linalg.lstsq(Zl, y, rcond=None)[0]; print('  linear AR with debt:', np.round(bl, 4), 'SSR', np.sum((y - Zl @ bl) ** 2))
    out = dict(cs=cs, ssr=ssr, c=c, th=th)
    if lab == 'inflation':
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(cs, ssr / len(y), 'k', lw=1.3); ax[0].axvline(c, color='k', ls=':', lw=0.8); ax[0].set_xlabel('Kink point $c$ (debt/GDP, %)'); ax[0].set_ylabel('Concentrated $S_n(c)$'); ax[0].set_title('(a) Concentrated least squares criterion', fontsize=10)
        xg = np.linspace(0, 121, 300); mg = th[0] * np.minimum(xg - c, 0) + th[1] * np.maximum(xg - c, 0) + th[2] * y1.mean() + th[3]
        ax[1].scatter(x, y, s=6, color='0.6'); ax[1].plot(xg, mg, 'k', lw=1.4); ax[1].plot([c], [th[2] * y1.mean() + th[3]], 'ks', ms=5)
        ax[1].set_xlabel('Lagged debt/GDP (%)'); ax[1].set_ylabel('Inflation (%)'); ax[1].set_title('(b) Regression kink fit (lagged inflation at its mean)', fontsize=10); ax[1].set_ylim(-16, 25)
        plt.tight_layout(); plt.savefig(R + 'ch23_09.pdf'); plt.close()
    # profile-based check of a flat criterion
    print('  criterion range: min %.2f max %.2f; c values within 1%% of min: [%.1f, %.1f]' % (ssr.min(), ssr.max(), cs[ssr <= 1.01 * ssr.min()].min(), cs[ssr <= 1.01 * ssr.min()].max()))

# ---------------- Nerlove smooth threshold
nd = pd.read_stata('Nerlove1963/Nerlove1963.dta')
lc = np.log(nd.cost.values); lq = np.log(nd.output.values); lp = np.log(nd.Plabor.values) + np.log(nd.Pcapital.values) + np.log(nd.Pfuel.values)
print('logQ quantiles', np.round(np.quantile(lq, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]), 3), 'sorted 10-15th smallest/largest', np.round(np.sort(lq)[[9, 14]], 3), np.round(np.sort(lq)[[-15, -10]], 3))
def mst(t):
    b1, b2, b3, b4, g = t
    return b1 + b2 * lq + b3 * lp + b4 * lq / (1 + np.exp(-(lq - g)))
# linear model first
Zl = np.column_stack([np.ones_like(lq), lq, lp]); bl = np.linalg.lstsq(Zl, lc, rcond=None)[0]; print('linear', bl, 'SSR', np.sum((lc - Zl @ bl) ** 2))
glo, ghi = np.sort(lq)[12], np.sort(lq)[-13]
print('gamma range', glo, ghi)
# (b) global search: multi-start least squares over all 5 parameters
best = None
for g0 in np.linspace(glo, ghi, 15):
    for b40 in [-0.5, 0, 0.5]:
        sol = least_squares(lambda t: lc - mst(t), [bl[0], bl[1], bl[2], b40, g0], xtol=1e-14, ftol=1e-14)
        if best is None or sol.cost < best.cost: best = sol
print('global NLLS', np.round(best.x, 4), 'SSR %.5f' % (2 * best.cost))
# (c) concentrated over gamma
gs = np.linspace(glo, ghi, 4001); ss = []
for g in gs:
    Z = np.column_stack([np.ones_like(lq), lq, lp, lq / (1 + np.exp(-(lq - g)))]); b = np.linalg.lstsq(Z, lc, rcond=None)[0]; ss.append(np.sum((lc - Z @ b) ** 2))
ss = np.array(ss); g = gs[ss.argmin()]
Z = np.column_stack([np.ones_like(lq), lq, lp, lq / (1 + np.exp(-(lq - g)))]); b = np.linalg.lstsq(Z, lc, rcond=None)[0]
thc = np.r_[b, g]; print('concentrated', np.round(thc, 4), 'SSR %.5f' % ss.min())
# refine
sol = least_squares(lambda t: lc - mst(t), thc, xtol=1e-15, ftol=1e-15); th = sol.x
V, e = nlls_se(mst, th, lc); se = np.sqrt(np.diag(V))
n = len(lc); V0 = np.linalg.inv(jac(mst, th).T @ jac(mst, th)) * (e @ e / (n - 5)); se0 = np.sqrt(np.diag(V0))
print('final', ' '.join('%.4f(%.4f|%.4f)' % (t, s, s0) for t, s, s0 in zip(th, se, se0)), 'SSR %.5f' % (e @ e))
print('slope low %.3f, high %.3f' % (th[1], th[1] + th[3]))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(gs, ss, 'k', lw=1.3); ax[0].axvline(g, color='k', ls=':', lw=0.8); ax[0].set_xlabel(r'$\gamma$'); ax[0].set_ylabel('Concentrated SSR'); ax[0].set_title(r'(a) Concentrated SSR as a function of $\gamma$', fontsize=10)
qg = np.linspace(lq.min(), lq.max(), 300)
ax[1].scatter(lq, lc - th[2] * lp, s=8, color='0.6'); ax[1].plot(qg, th[0] + th[1] * qg + th[3] * qg / (1 + np.exp(-(qg - th[4]))), 'k', lw=1.4)
ax[1].set_xlabel('log Q'); ax[1].set_ylabel(r'log TC $-\hat\beta_3$(log prices)'); ax[1].set_title('(b) Fitted smooth threshold cost function', fontsize=10)
plt.tight_layout(); plt.savefig(R + 'ch23_10.pdf'); plt.close()
