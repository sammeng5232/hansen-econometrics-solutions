import numpy as np, pandas as pd
from urlib import *
q=pd.read_stata('progs/FRED-QD.dta'); md=pd.read_stata('progs/FRED-MD.dta')
def run(y, p, M, freq):
    r=adf(y,p,'ct'); pv=interp_p(r['t'],ADF_CV['ct'],PCT)
    k=kpss(y,M,'ct'); kp=interp_p(k,KPSS_CV['ct'],PCT,lower_tail=False)
    return '%.3f (%.3f) ADF %.2f p %s | KPSS2 %.3f p %s'%(r['rho'],r['se'],r['t'],fmt_p(pv),k,fmt_p(kp))
y=np.log(q['gdpc1'].values); 
# AIC on sample from 1961q1 (index 8)
print('gdp aic p', aic_ar(y,8,'ct',start=8)[0], run(y,3,18,'q'))
y=np.log(q['pcndx'].values); print('cons aic p', aic_ar(y,8,'ct',start=8)[0], run(y,4,18,'q'))
for nm,f,pp in [('excausx',np.log,11),('gs10',lambda x:x,12),('oilpricex',np.log,2),('unrate',lambda x:x,7),('cpiaucsl',np.log,11),('sp500',np.log,6)]:
    y=f(md[nm].values.astype(float)); ok=~np.isnan(y); y=y[ok]
    print(nm, 'aic p', aic_ar(y,12,'ct',start=24)[0], run(y,pp,26,'m'))
print('unrate KPSS1', kpss(md['unrate'].values.astype(float),26,'c'))
# spread & Johansen quarterly
Y=q[['tb3ms','gs10']].values.astype(float)
print('var aic', aic_var_levels(Y,8)[0])
J=johansen(Y,4,2); print('LR',J['LR'],'beta',J['beta'], J['n'])
u=ols(Y[:,0], np.column_stack([np.ones(len(Y)),Y[:,1]]))
print('EG coef',u[0]); print('EG', adf(u[2],8,'c'))
Y5=q[['tb3ms','tb6ms','gs1','gs5','gs10']].values.astype(float)
print('5 rates', johansen(Y5,4,2)['LR'])
