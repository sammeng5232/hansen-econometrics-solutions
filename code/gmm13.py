import numpy as np, pandas as pd
from ivlib import tsls, ols
from scipy import stats
def gmm2(y,X,Z,label,names):
    n=len(y)
    b0,V0,e0=tsls(y,X,Z,robust=True)
    Om=(Z.T@(Z*(e0**2)[:,None]))/n; W=np.linalg.inv(Om)
    A=X.T@Z@W@Z.T@X; b=np.linalg.solve(A,X.T@Z@W@Z.T@y); e=y-X@b
    g=Z.T@e/n; J1=n*g@W@g
    Om2=(Z.T@(Z*(e**2)[:,None]))/n; W2=np.linalg.inv(Om2); J2=n*g@W2@g
    Q=Z.T@X/n; V=np.linalg.inv(Q.T@W2@Q)/n
    df=Z.shape[1]-X.shape[1]
    print('---',label)
    for i,nm in enumerate(names): print(f'   {nm:10s} 2SLS {b0[i]: .4f} ({np.sqrt(V0[i,i]):.4f})   GMM {b[i]: .4f} ({np.sqrt(V[i,i]):.4f})')
    print(f'   J (first-step weight) {J1:.3f}  J (final weight) {J2:.3f}  df {df}  p {stats.chi2.sf(J1,df):.4f}')
d=pd.read_stata('AJR2001/AJR2001.dta'); n=len(d); one=np.ones(n)
y=d.loggdp.values.astype(float); r=d.risk.values.astype(float); lm=d.logmort0.values.astype(float)
gmm2(y,np.column_stack([r,one]),np.column_stack([lm,lm**2,one]),'13.27 AJR',['risk','const'])
c=pd.read_stata('Card1995/Card1995.dta'); c=c.astype({k:float for k in ['ed76','age76','black','reg76r','smsa76r','nearc4a','nearc4b','lwage76']}); c=c[c.lwage76.notna()]; n=len(c); one=np.ones(n)
yw=c.lwage76.values; ed=c.ed76.values; age=c.age76.values; ex=age-ed-6; pub=c.nearc4a.values; pri=c.nearc4b.values
exo=np.column_stack([ex,ex**2/100,c.black,c.reg76r,c.smsa76r,one]); X=np.column_stack([ed,exo])
nm=['education','experience','exp2/100','black','south','urban','const']
gmm2(yw,X,np.column_stack([exo,pub,pri]),'13.28(a) Card 2SLS(a) spec',nm)
gmm2(yw,X,np.column_stack([exo,pub,pri,pub*age,pub*age**2/100]),'13.28(b) Card expanded spec',nm)
