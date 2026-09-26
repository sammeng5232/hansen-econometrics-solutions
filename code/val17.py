import numpy as np, pandas as pd, time
from dpd2 import dpd, show
ab=pd.read_stata('AB1991/AB1991.dta'); ab['id']=ab['id'].astype(int); ab['year']=ab['year'].astype(int)
regs=[('n',1),('w',0),('w',1),('k',0),('k',1)]
t0=time.time(); r=dpd(ab,'id','year','n',regs,[('n',2,None),('w',2,None),('k',2,None)]); print(show(r), r['nd'], r['L'], '%.1fs'%(time.time()-t0))
iv=pd.read_stata('Invest1993/Invest1993.dta'); iv['cusip']=iv['cusip'].astype(int); iv['year']=iv['year'].astype(int)
t0=time.time(); r=dpd(iv,'cusip','year','inva',[('inva',1),('inva',2)],[('inva',2,6)],system=True,twostep=True); print(show(r,2), r['nd'], r['nl'], r['L'], '%.1fs'%(time.time()-t0))
regs=[('inva',1),('inva',2),('vala',1),('vala',2),('debta',1),('debta',2),('cfa',1),('cfa',2)]
t0=time.time(); r=dpd(iv,'cusip','year','inva',regs,[(v,2,6) for v in ['inva','vala','debta','cfa']],system=True,twostep=True); print(show(r,2), r['L'], '%.1fs'%(time.time()-t0))
