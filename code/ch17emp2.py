import numpy as np, pandas as pd, time
from dpd2 import dpd, show
ab=pd.read_stata('AB1991/AB1991.dta'); ab['id']=ab['id'].astype(int); ab['year']=ab['year'].astype(int)
regs=[('n',1),('w',0),('w',1),('k',0),('k',1)]
for lab,sysm in [('AB',False),('BB',True)]:
    r=dpd(ab,'id','year','n',regs,[('n',2,None),('w',2,None),('k',2,None)],system=sysm)
    th=r['th1'][:5]; V=r['V1'][:5,:5]; a=th[0]
    for nm,idx in [('wage',(1,2)),('capital',(3,4))]:
        b=th[idx[0]]+th[idx[1]]; lr=b/(1-a)
        g=np.zeros(5); g[0]=b/(1-a)**2; g[idx[0]]=1/(1-a); g[idx[1]]=1/(1-a)
        print(lab,nm,'LR %.3f (%.3f)'%(lr,np.sqrt(g@V@g)))
iv=pd.read_stata('Invest1993/Invest1993.dta'); iv['cusip']=iv['cusip'].astype(int); iv['year']=iv['year'].astype(int)
print('=== 17.17 debt AR ===')
for p in [1,2]:
  regs=[('debta',j) for j in range(1,p+1)]
  for lags in [None,6,3]:
    for sysm in [False,True]:
      for td in [False,True]:
        t0=time.time()
        r=dpd(iv,'cusip','year','debta',regs,[('debta',2,lags)],system=sysm,twostep=True,timedum=td)
        lab='%s AR(%d) lags2-%s td=%d'%('BB' if sysm else 'AB',p,lags,td)
        print(lab,'| 1step',' '.join('%.4f(%.4f|c%.4f)'%(r['th1'][j],r['se1'][j],r['se1c'][j]) for j in range(p)),
              '| 2step',' '.join('%.4f(%.4f|c%.4f)'%(r['th2'][j],r['se2'][j],r['se2c'][j]) for j in range(p)),'L=%d J=%.1f nd=%d %.0fs'%(r['L'],r['J'],r['nd'],time.time()-t0),flush=True)
