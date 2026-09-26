import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from felib import *
ck=pd.read_stata('CK1994/CK1994.dta')
ck['fte']=ck.empft+ck.emppt/2+ck.nmgrs
c=ck.dropna(subset=['fte']).copy(); c=c[c.groupby('store')['fte'].transform('size')==2]
print('n',len(c), c.store.nunique())
print(c.groupby(['time','state'])['fte'].mean().unstack())
c['treatment']=c.time*c.state
r=ols_cluster(c.fte,np.column_stack([np.ones(len(c)),c.state,c.time,c.treatment]),c.store,['c','state','time','D']); print('(18.2)',show(r,2))
r=xtreg_fe(c,'fte',['treatment','time'],'store'); print('(18.4)',show(r,2))
r=xtreg_fe(c,'fte',['treatment','time','hoursopen'],'store'); print('hours',show(r,2))
for reg in ['southj','centralj','northj','pa1','pa2']:
    print(reg, c[c[reg]==1].groupby('time')['fte'].mean().round(1).tolist())
for v in ['southj','northj','pa1']: c['treat_'+v]=c.time*c[v]
r=xtreg_fe(c,'fte',['treatment','time','treat_southj','treat_northj'],'store'); print(show(r,2), 'test p', wald(r,[2,3]))
r=xtreg_fe(c,'fte',['treatment','time','treat_pa1'],'store'); print(show(r,2), 'test', wald(r,[2]))
ds=pd.read_stata('DS2004/DS2004.dta'); ds=ds[ds.month!=7].copy(); ds['after']=(ds.month>7).astype(float); ds['treatment']=ds.sameblock*ds.after
print(ds.groupby(['after','sameblock'])['thefts'].mean().unstack().round(3))
md=pd.get_dummies(ds.month,prefix='m',drop_first=True).astype(float); ds=pd.concat([ds,md],axis=1)
r=xtreg_fe(ds,'thefts',['treatment']+md.columns.tolist(),'block'); print('DS',show(r,4)[:40], r['n'])
bm=pd.read_stata('BMN2016/BMN2016.dta')
yd=pd.get_dummies(bm.year,prefix='y',drop_first=True).astype(float); bm=pd.concat([bm,yd],axis=1)
xs=['liqonsun','liqoffsun','unempw','liqOnOutflows','liqOffOutflows']
r=xtreg_fe(bm,'logliq',xs+yd.columns.tolist(),'id'); print('(18.7)',show(r,3)[:200], r['n'], r['G'])
for i in bm.id.unique():
    bm['tr%d'%int(i)]=(bm.id==i)*bm.year
trc=['tr%d'%int(i) for i in bm.id.unique()]
r=xtreg_fe(bm,'logliq',xs+yd.columns.tolist()+trc,'id'); print('(18.8)',show(r,3)[:200])
