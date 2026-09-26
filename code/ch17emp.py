import numpy as np, pandas as pd, time
from dpd2 import dpd, show
ab=pd.read_stata('AB1991/AB1991.dta'); ab['id']=ab['id'].astype(int); ab['year']=ab['year'].astype(int)
print('=== 17.15 capital AR(1) ===')
r=dpd(ab,'id','year','k',[('k',1)],[('k',2,None)]); print('AB1', show(r), 'classical se %.4f'%r['se1c'][0], r['nd'], r['L'])
r=dpd(ab,'id','year','k',[('k',1)],[('k',2,None)],system=True); print('BB1', show(r), r['nd'], r['nl'], r['L'])
# pooled OLS and FE for bracketing
import numpy.linalg as la
g=ab.sort_values(['id','year']).copy(); g['kl']=g.groupby('id')['k'].shift(1)
h=g.dropna(subset=['kl']); yd=pd.get_dummies(h['year'],drop_first=True).astype(float).values
X=np.column_stack([np.ones(len(h)),h['kl'],yd]); b=la.lstsq(X,h['k'],rcond=None)[0]; print('pooled OLS', b[1])
dm=lambda s: s-h.groupby('id')[s.name].transform('mean')
Xw=np.column_stack([dm(h['kl'])]+[yd[:,j]-pd.Series(yd[:,j],index=h.index).groupby(h['id']).transform('mean').values for j in range(yd.shape[1])]); bw=la.lstsq(Xw,dm(h['k']),rcond=None)[0]; print('FE', bw[0])
# first-stage strength: regress dk_{t-1} on k_{t-2} (Anderson-Hsiao)
g['dkl']=g['kl']-g.groupby('id')['k'].shift(2); g['kl2']=g.groupby('id')['k'].shift(2); q=g.dropna(subset=['dkl','kl2'])
bb=la.lstsq(np.column_stack([np.ones(len(q)),q['kl2']]),q['dkl'],rcond=None)[0]; print('AH first stage gamma', bb[1])
print('=== 17.16 employment ===')
regs=[('n',1),('w',0),('w',1),('k',0),('k',1)]
r=dpd(ab,'id','year','n',regs,[('n',2,None)],iv_exog=[('w',0),('w',1),('k',0),('k',1)]); print('(a) strict exog', show(r), r['L'])
r=dpd(ab,'id','year','n',regs,[('n',2,None),('w',2,None),('k',2,None)]); print('(b) AB predet', show(r), r['L'])
rb=dpd(ab,'id','year','n',regs,[('n',2,None),('w',2,None),('k',2,None)],system=True); print('(c) BB', show(rb), rb['L'])
print('(e) BB classical se', np.round(rb['se1c'][:5],4), 'ratio robust/classical', np.round(rb['se1'][:5]/rb['se1c'][:5],2))
for lab,rr in [('AB',r),('BB',rb)]:
    th=rr['th1']; a=th[0]; print(lab,'LR wage %.3f LR capital %.3f'%((th[1]+th[2])/(1-a),(th[3]+th[4])/(1-a)))
