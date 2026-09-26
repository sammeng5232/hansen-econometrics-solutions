import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from felib import *
ck=pd.read_stata('CK1994/CK1994.dta')
ck['price']=ck.priceentree+ck.pricefry+ck.pricesoda
print('missing price', ck.price.isna().sum(), 'of', len(ck))
c=ck.dropna(subset=['price']).copy(); c=c[c.groupby('store')['price'].transform('size')==2].copy()
print('balanced n',len(c),'stores',c.store.nunique(), 'NJ', c[c.state==1].store.nunique(),'PA',c[c.state==0].store.nunique())
print(c.price.describe())
t=c.groupby(['time','state'])['price'].mean().unstack(); print(t.round(3)); print('DiD', (t.loc[1,1]-t.loc[0,1])-(t.loc[1,0]-t.loc[0,0]))
c['treatment']=c.time*c.state
r=ols_cluster(c.price,np.column_stack([np.ones(len(c)),c.state,c.time,c.treatment]),c.store,['c','state','time','D']); print('(c)',show(r,3))
# (d) state FE, classical se
W=demean(c,['price','treatment','time'],'state'); X=W[['treatment','time']].values; y=W.price.values
b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; n=len(y); k=2+2; s2=e@e/(n-k); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X))); print('(d) state FE classical',b.round(3),se.round(3))
rs=ols_cluster(y,X,c.store.values,['D','time'],dfk=3); print('(d) state FE, store-clustered', show(rs,3))
r=xtreg_fe(c,'price',['treatment','time'],'store'); print('(e) store FE robust',show(r,3))
for reg in ['southj','centralj','northj','pa1','pa2']:
    m=c[c[reg]==1].groupby('time')['price'].mean(); print(reg, round(m[0],3), round(m[1],3), round(m[1]-m[0],3), c[(c[reg]==1)].store.nunique())
for v in ['southj','northj','pa1']: c['treat_'+v]=c.time*c[v]
r=xtreg_fe(c,'price',['treatment','time','treat_southj','treat_northj'],'store'); print('(h)',show(r,3),'W,p,F,pF',np.round(wald(r,[2,3]),3))
r=xtreg_fe(c,'price',['treatment','time','treat_pa1'],'store'); print('(i)',show(r,3),'W,p',np.round(wald(r,[2]),3))
# components
for comp in ['priceentree','pricefry','pricesoda']:
    cc=ck.dropna(subset=[comp]).copy(); cc=cc[cc.groupby('store')[comp].transform('size')==2].copy(); cc['treatment']=cc.time*cc.state
    r=xtreg_fe(cc,comp,['treatment','time'],'store'); print(comp, show(r,3), len(cc))
# log price
c['lp']=np.log(c.price); r=xtreg_fe(c,'lp',['treatment','time'],'store'); print('log price',show(r,4))
print('pre mean NJ', c[(c.state==1)&(c.time==0)].price.mean())
print('=== DS oneblock ===')
ds=pd.read_stata('DS2004/DS2004.dta'); ds=ds[ds.month!=7].copy(); ds['after']=(ds.month>7).astype(float)
print('blocks', ds.block.nunique(), 'sameblock',ds[ds.sameblock==1].block.nunique(),'oneblock',ds[ds.oneblock==1].block.nunique(), 'overlap', ds[(ds.sameblock==1)&(ds.oneblock==1)].block.nunique())
print(ds.groupby('distance').block.nunique())
sub=ds[ds.sameblock==0]
t=sub.groupby(['after','oneblock'])['thefts'].mean().unstack(); print(t.round(4)); print('DiD', (t.loc[1,1]-t.loc[0,1])-(t.loc[1,0]-t.loc[0,0]))
ds['treat_same']=ds.sameblock*ds.after; ds['treat_one']=ds.oneblock*ds.after
md=pd.get_dummies(ds.month,prefix='m',drop_first=True).astype(float); ds=pd.concat([ds,md],axis=1)
r=xtreg_fe(ds,'thefts',['treat_same','treat_one']+md.columns.tolist(),'block'); print('(b)',show(r,4)[:80], 'n',r['n'],'G',r['G'])
print('diff same-one', r['b'][0]-r['b'][1], np.sqrt(r['V'][0,0]+r['V'][1,1]-2*r['V'][0,1]))
# two blocks away
if 'distance' in ds:
    ds['twob']=(ds.distance==2).astype(float); ds['treat_two']=ds.twob*ds.after
    r=xtreg_fe(ds,'thefts',['treat_same','treat_one','treat_two']+md.columns.tolist(),'block'); print('with two',show(r,4)[:120])
# oneblock pre-period by month
print(ds.groupby(['month','oneblock'])['thefts'].mean().unstack().round(3))
