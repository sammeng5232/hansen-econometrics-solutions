import numpy as np, pandas as pd, time
from dpd2 import dpd, show
iv=pd.read_stata('Invest1993/Invest1993.dta'); iv['cusip']=iv['cusip'].astype(int); iv['year']=iv['year'].astype(int)
def fmt(r,step,p):
    th=r['th%d'%step]; se=r['se%d'%step]; sc=r['se%dc'%step]
    return ' '.join('%.3f(%.3f|%.3f)'%(th[j],se[j],sc[j]) for j in range(p))
print('=== 17.17 debt AR (no time effects) ===')
for p in [1,2]:
  regs=[('debta',j) for j in range(1,p+1)]
  for lags in [None,6,3]:
    for sysm in [False,True]:
      r=dpd(iv,'cusip','year','debta',regs,[('debta',2,lags)],system=sysm,twostep=True,timedum=False)
      print('%s AR(%d) lags2-%s | 1step %s | 2step %s | L=%d J=%.0f n=%d'%('BB' if sysm else 'AB',p,lags,fmt(r,1,p),fmt(r,2,p),r['L'],r['J'],r['nd']),flush=True)
r=dpd(iv,'cusip','year','debta',[('debta',1)],[('debta',2,None)],system=False,twostep=True,timedum=True); print('AB AR1 all, time effects',fmt(r,2,1))
r=dpd(iv,'cusip','year','debta',[('debta',1)],[('debta',2,None)],system=True,twostep=True,timedum=True); print('BB AR1 all, time effects',fmt(r,2,1))
print('=== 17.18 ===')
regs=[('debta',1),('inva',1),('vala',1),('cfa',1)]
for lags in [None,6,3]:
  for sysm in [False,True]:
    t0=time.time()
    r=dpd(iv,'cusip','year','debta',regs,[(v,2,lags) for v in ['debta','inva','vala','cfa']],system=sysm,twostep=True,timedum=False)
    print('%s lags2-%s | 1step %s | 2step %s | L=%d J=%.0f %.0fs'%('BB' if sysm else 'AB',lags,fmt(r,1,4),fmt(r,2,4),r['L'],r['J'],time.time()-t0),flush=True)
