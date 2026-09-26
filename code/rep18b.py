import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from felib import *
bm=pd.read_stata('BMN2016/BMN2016.dta')
yd=pd.get_dummies(bm.year,prefix='y',drop_first=True).astype(float); bm=pd.concat([bm,yd],axis=1)
ids=sorted(bm.id.unique())
for i in ids: bm['tr%d'%int(i)]=(bm.id==i)*(bm.year-1970.0)
trc=['tr%d'%int(i) for i in ids][1:]
for pre in ['liq','beer','wine']:
    xs=[pre+'onsun',pre+'offsun','unempw',pre+'OnOutflows',pre+'OffOutflows']
    r=xtreg_fe(bm,'log'+pre,xs+yd.columns.tolist(),'id'); print(pre,'(18.7)',show(r,4)[:230], r['n'], r['G'], 'year test', wald(r,list(range(5,5+yd.shape[1])))[1])
    r=xtreg_fe(bm,'log'+pre,xs+yd.columns.tolist()+trc,'id'); print(pre,'(18.8)',show(r,4)[:230], r['n'])
