import os
import numpy as np, pandas as pd, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from kreg import *
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', '')
res = {}


def analyze(name, x, y, lo, hi, cl=None, methods=('NW', 'LL'), ngrid=101, cvcl=False, nh=61):
    n = len(y); h0 = hrot(x, y, lo, hi); hs = np.linspace(h0 / 3, 3 * h0, nh)
    xg = np.linspace(lo, hi, ngrid); out = {'n': n, 'hrot': h0}
    for m in methods:
        c = cv(x, y, hs, m, cl if cvcl else None, lo, hi); i = c.argmin(); hcv = hs[i]; edge = (i == 0 or i == len(hs) - 1)
        h = h0 if edge else hcv
        e = y - loo(x, y, h, m, cl if cvcl else None)
        mh = fit(xg, x, y, h, m); s = se(xg, x, e, h, m, cl); s0 = se(xg, x, e, h, m, None) if cl is not None else s
        out[m] = dict(hcv=hcv, edge=bool(edge), h=h, xg=xg, m=mh, se=s, se0=s0)
        print(name, m, 'n', n, 'hrot %.3f hcv %.3f%s used %.3f' % (h0, hcv, ' (edge)' if edge else '', h), flush=True)
    out['lin'] = np.polyfit(x, y, 1)
    return out


def summ(o, m, pts):
    d = o[m]
    return {p: (round(float(np.interp(p, d['xg'], d['m'])), 3), round(float(np.interp(p, d['xg'], d['se'])), 3)) for p in pts}


def band(a, d, col='0.82', lab=None, lc='k'):
    a.fill_between(d['xg'], d['m'] - 1.96 * d['se'], d['m'] + 1.96 * d['se'], color=col, lw=0)
    a.plot(d['xg'], d['m'], color=lc, lw=1.4, label=lab)


TITLE = {'NW': '(a) Nadaraya-Watson', 'LL': '(b) Local linear'}

# ---- 19.7 DDK
d = pd.read_stata('DDK2011/DDK2011.dta')
for lab, gv in [('boys', 0), ('girls', 1)]:
    g = d[(d.tracking == 1) & (d.girl == gv)].dropna(subset=['totalscore', 'percentile'])
    x = g.percentile.values.astype(float); y = g.totalscore.values.astype(float); cl = pd.factorize(g.schoolid)[0]
    o = analyze('DDK ' + lab, x, y, 0, 100, cl=cl, methods=('LL',), cvcl=True)
    hs = np.linspace(4, 20, 81); c1 = cv(x, y, hs, 'LL'); print(lab, 'conventional CV h', hs[c1.argmin()], 'schools', cl.max() + 1)
    res['ddk_' + lab] = o
    print(summ(o, 'LL', [0, 10, 25, 50, 75, 80, 90, 100]), 'lin', o['lin'])
    print('se ratio clustered/unclustered mean', np.mean(o['LL']['se'] / o['LL']['se0']))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, lab in zip(ax, ['boys', 'girls']):
    o = res['ddk_' + lab]['LL']; band(a, o, lab='local linear')
    a.plot(o['xg'], np.polyval(res['ddk_' + lab]['lin'], o['xg']), 'k--', lw=1, label='global linear')
    a.set_title('Tracked %s (h = %.1f)' % (lab, o['h']), fontsize=10); a.set_xlabel('Initial percentile'); a.set_xlim(0, 100)
ax[0].set_ylabel('Test score'); ax[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(R + 'ch19_07.pdf'); plt.close()

# ---- 19.8 CPS education = 20
c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c['exp'] = c[0] - c[3] - 6; c['lw'] = np.log(c[4] / (c[5] * c[6]))
s = c[(c[3] == 20) & (c['exp'] >= 0) & (c['exp'] <= 40)]
for lab, f in [('men', 0), ('women', 1)]:
    g = s[s[1] == f]; x = g['exp'].values.astype(float); y = g['lw'].values
    o = analyze('CPS ' + lab, x, y, 0, 40); res['cps_' + lab] = o
    for m in ['NW', 'LL']:
        print(lab, m, summ(o, m, [0, 5, 10, 15, 20, 25, 30, 35, 40]))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, m in zip(ax, ['NW', 'LL']):
    for lab, col in [('men', 'k'), ('women', '0.45')]:
        o = res['cps_' + lab][m]
        a.fill_between(o['xg'], o['m'] - 1.96 * o['se'], o['m'] + 1.96 * o['se'], color=col, alpha=0.18, lw=0)
        a.plot(o['xg'], o['m'], color=col, lw=1.4, label='%s (h = %.1f)' % (lab, o['h']))
    a.set_title(TITLE[m], fontsize=10); a.set_xlabel('Experience (years)'); a.set_xlim(0, 40); a.legend(frameon=False, fontsize=8)
ax[0].set_ylabel('Log wage'); plt.tight_layout(); plt.savefig(R + 'ch19_08.pdf'); plt.close()

# ---- 19.9 Invest1993, Q <= 5
iv = pd.read_stata('Invest1993/Invest1993.dta'); iv = iv[iv.vala <= 5]
x = iv.vala.values.astype(float); y = iv.inva.values.astype(float); cl = pd.factorize(iv.cusip)[0]
t0 = time.time(); o = analyze('Invest', x, y, 0, 5, cl=cl, nh=13); res['inv'] = o; print('time %.0f' % (time.time() - t0))
for m in ['NW', 'LL']:
    print('inv', m, summ(o, m, [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5]), 'se ratio cl/iid', np.mean(o[m]['se'] / o[m]['se0']))
print('n', len(y), 'quantiles Q', np.quantile(x, [.05, .25, .5, .75, .95]))
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, m in zip(ax, ['NW', 'LL']):
    d_ = o[m]; band(a, d_, lab='kernel estimate (h = %.2f)' % d_['h'])
    a.plot(d_['xg'], np.polyval(o['lin'], d_['xg']), 'k--', lw=1, label='global linear')
    a.set_title(TITLE[m], fontsize=10); a.set_xlabel('Q'); a.set_xlim(0, 5); a.legend(frameon=False, fontsize=8)
ax[0].set_ylabel('I (investment/assets)'); plt.tight_layout(); plt.savefig(R + 'ch19_09.pdf'); plt.close()

# ---- 19.10 RR2010
rr = pd.read_stata('RR2010/RR2010.dta')
x = rr.debt.values.astype(float); y = rr.gdp.values.astype(float)
print('debt range', x.min(), x.max(), 'n>90', (x > 90).sum(), rr.year[x > 90].tolist())
o2 = analyze('RR debt', x, y, 0, 121); res['rr'] = o2
for m in ['NW', 'LL']:
    print('rr', m, summ(o2, m, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
xi = rr.inflation.values.astype(float); lo2, hi2 = np.quantile(xi, [0.025, 0.975]); print('infl range', xi.min(), xi.max(), lo2, hi2)
o3 = analyze('RR inflation', xi, y, lo2, hi2); res['rr_inf'] = o3
for m in ['NW', 'LL']:
    print('rr_inf', m, summ(o3, m, [-10, -5, -2, 0, 2, 5, 10, 15]))
print('debt>90 growth', y[x > 90].mean(), 'debt<=90', y[x <= 90].mean(), 'lin', o2['lin'], 'lin infl', o3['lin'])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, m in zip(ax[:2], ['NW', 'LL']):
    d_ = o2[m]; a.scatter(x, y, s=6, color='0.6')
    a.fill_between(d_['xg'], d_['m'] - 1.96 * d_['se'], d_['m'] + 1.96 * d_['se'], color='0.8', alpha=0.8)
    a.plot(d_['xg'], d_['m'], 'k', lw=1.4, label='h = %.1f' % d_['h'])
    a.axvline(90, color='k', ls=':', lw=0.8); a.set_title(('(a) NW' if m == 'NW' else '(b) LL') + ': growth on debt/GDP', fontsize=10)
    a.set_xlabel('Debt/GDP (%)'); a.set_ylim(-15, 20); a.legend(frameon=False, fontsize=8)
d_ = o3['LL']; ax[2].scatter(xi, y, s=6, color='0.6')
ax[2].fill_between(d_['xg'], d_['m'] - 1.96 * d_['se'], d_['m'] + 1.96 * d_['se'], color='0.8', alpha=0.8)
ax[2].plot(d_['xg'], d_['m'], 'k', lw=1.4, label='LL, h = %.1f' % d_['h'])
d_ = o3['NW']; ax[2].plot(d_['xg'], d_['m'], 'k--', lw=1, label='NW, h = %.1f' % d_['h'])
ax[2].set_title('(c) Growth on inflation', fontsize=10); ax[2].set_xlabel('Inflation (%)'); ax[2].set_ylim(-15, 20); ax[2].legend(frameon=False, fontsize=8)
ax[0].set_ylabel('GDP growth (%)'); plt.tight_layout(); plt.savefig(R + 'ch19_10.pdf'); plt.close()

# ---- 19.11 nonlinear AR(1) for GDP growth
q = pd.read_stata('progs/FRED-QD.dta'); gdp = q.gdpc1.values.astype(float)
Y = 100 * ((gdp[1:] / gdp[:-1]) ** 4 - 1); x = Y[:-1]; y = Y[1:]
lo, hi = np.quantile(x, [0.025, 0.975]); print('gdp n', len(y), 'range', x.min(), x.max(), lo, hi, 'mean', Y.mean())
o = analyze('GDP AR', x, y, lo, hi); res['gdp'] = o
for m in ['NW', 'LL']:
    print('gdp', m, summ(o, m, [-2, 0, 2, 3, 4, 6, 8]))
print('lin', o['lin'])
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, m in zip(ax, ['NW', 'LL']):
    d_ = o[m]; a.scatter(x, y, s=6, color='0.6')
    a.fill_between(d_['xg'], d_['m'] - 1.96 * d_['se'], d_['m'] + 1.96 * d_['se'], color='0.8', alpha=0.8)
    a.plot(d_['xg'], d_['m'], 'k', lw=1.4, label='kernel (h = %.2f)' % d_['h'])
    a.plot(d_['xg'], np.polyval(o['lin'], d_['xg']), 'k--', lw=1, label='linear AR(1)')
    a.set_title(TITLE[m], fontsize=10); a.set_xlabel('Lagged growth (annualized, %)'); a.set_xlim(lo, hi); a.set_ylim(-5, 12); a.legend(frameon=False, fontsize=8)
ax[0].set_ylabel('GDP growth (%)'); plt.tight_layout(); plt.savefig(R + 'ch19_11.pdf'); plt.close()
