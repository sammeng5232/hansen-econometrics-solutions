import numpy as np, pandas as pd
from urlib import *
md=pd.read_stata('progs/FRED-MD.dta')
ser=[('log(rpi)',np.log(md['rpi'].values.astype(float))),
     ('indpro',md['indpro'].values.astype(float)),
     ('log(indpro)',np.log(md['indpro'].values.astype(float))),
     ('houst',md['houst'].values.astype(float)),
     ('hwi',md['hwi'].values.astype(float)),
     ('clf16ov',md['clf16ov'].values.astype(float)),
     ('log(clf16ov)',np.log(md['clf16ov'].values.astype(float))),
     ('claimsx',md['claimsx'].values.astype(float)),
     ('log(claimsx)',np.log(md['claimsx'].values.astype(float))),
     ('ipfuels',md['ipfuels'].values.astype(float)),
     ('log(ipfuels)',np.log(md['ipfuels'].values.astype(float)))]
T=len(md); print('T',T, 'M rule', 3*T**(1/3))
print('=== ADF ===')
for nm,y in ser:
    assert not np.isnan(y).any()
    row=[nm]
    for det in ['c','ct']:
        p,_=aic_ar(y,12,det,start=12)
        r=adf(y,p,det); pv=interp_p(r['t'],ADF_CV[det],PCT)
        row.append('%s p=%d rho-1=%.4f(%.4f) ADF=%.2f pv=%s'%(det,p,r['rho'],r['se'],r['t'],fmt_p(pv)))
    print(' | '.join(row))
print('=== KPSS ===')
for nm,y in ser:
    row=[nm]
    for det in ['c','ct']:
        for M in [26,12,6]:
            k=kpss(y,M,det); kp=interp_p(k,KPSS_CV[det],PCT,lower_tail=False)
            row.append('%s M=%d %.3f(%s)'%(det,M,k,fmt_p(kp)))
    print(' | '.join(row))
print('=== Johansen ===')
pairs=[('tb3ms,gs10',md[['tb3ms','gs10']].values.astype(float),[2,3]),
       ('aaa,baa',md[['aaa','baa']].values.astype(float),[2,3]),
       ('log ipd, ipn',np.log(md[['ipdcongd','ipncongd']].values.astype(float)),[3,4])]
for nm,Y,models in pairs:
    for tr in [False,True]:
        p,a=aic_var_levels(Y,12,trend=tr,start=12); print(nm,'AIC p (trend=%s)'%tr,p)
    pa,_=aic_var_levels(Y,12,trend=(models[0]==3 and nm.startswith('log')),start=12)
    for p in sorted(set([pa,4,12])):
        for mo in models:
            J=johansen(Y,p,mo)
            ps=[fmt_p(interp_p(J['LR'][r],JOH[mo][:,2-r-1],JPCT,lower_tail=False)) for r in range(2)]
            print('  p=%d model=%d n=%d LR0=%.2f (p %s) LR1=%.2f (p %s) beta=%s'%(p,mo,J['n'],J['LR'][0],ps[0],J['LR'][1],ps[1],np.round(J['beta'],3)))
