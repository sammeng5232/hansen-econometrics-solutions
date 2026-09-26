import numpy as np, pandas as pd
from scipy.stats import chi2
from dpd2 import dpd, show
ab=pd.read_stata('AB1991/AB1991.dta'); ab['id']=ab['id'].astype(int); ab['year']=ab['year'].astype(int)
r=dpd(ab,'id','year','k',[('k',1)],[('k',2,None)],system=False,twostep=True); print('17.15 AB J %.1f df %d p %.3f; 1step %s'%(r['J'],r['L']-len(r['th2']),chi2.sf(r['J'],r['L']-len(r['th2'])),show(r,1)))
r=dpd(ab,'id','year','k',[('k',1)],[('k',2,None)],system=True,twostep=True); print('17.15 BB J %.1f df %d p %.3f; 1step %s'%(r['J'],r['L']-len(r['th2']),chi2.sf(r['J'],r['L']-len(r['th2'])),show(r,1)))
# sigma_u/sigma_e ratio for capital via FE residuals
iv=pd.read_stata('Invest1993/Invest1993.dta'); iv['cusip']=iv['cusip'].astype(int); iv['year']=iv['year'].astype(int)
for lab,y,regs,dg,sysm in [('17.17a AB',  'debta',[('debta',1)],[('debta',2,None)],False),
                           ('17.17b BB',  'debta',[('debta',1)],[('debta',2,None)],True),
                           ('17.18a AB all','debta',[('debta',1),('inva',1),('vala',1),('cfa',1)],[(v,2,None) for v in ['debta','inva','vala','cfa']],False),
                           ('17.18a AB 2-6','debta',[('debta',1),('inva',1),('vala',1),('cfa',1)],[(v,2,6) for v in ['debta','inva','vala','cfa']],False),
                           ('17.18b BB all','debta',[('debta',1),('inva',1),('vala',1),('cfa',1)],[(v,2,None) for v in ['debta','inva','vala','cfa']],True),
                           ('17.18b BB 2-6','debta',[('debta',1),('inva',1),('vala',1),('cfa',1)],[(v,2,6) for v in ['debta','inva','vala','cfa']],True)]:
    r=dpd(iv,'cusip','year',y,regs,dg,system=sysm,twostep=True,timedum=False)
    df_=r['L']-len(r['th2'])
    print(lab,'2step',show(r,2),'| naive',np.round(r['se2c'][:len(regs)],4),'| J %.0f df %d p %.2g | N %d nd %d nl %d'%(r['J'],df_,chi2.sf(r['J'],df_),r['N'],r['nd'],r['nl']))
    print('     1step',show(r,1),'| classical',np.round(r['se1c'][:len(regs)],4))
