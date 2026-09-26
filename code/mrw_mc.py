import numpy as np, pandas as pd
from lib import ols, wald
from scipy import stats
md=pd.read_stata('MRW1992/MRW1992.dta'); md=md[md.N==1]; n=len(md)
yM=np.log(md.Y85.values)-np.log(md.Y60.values)
X=np.column_stack([np.log(md.Y60),np.log(md.invest/100),np.log(md.pop_growth/100+0.05),np.log(md.school/100),np.ones(n)])
b,V,e,_=ols(yM,X,'HC1')
R=np.array([[0,1,1,1,0.]]).T
s=R.T@b; se=np.sqrt(R.T@V@R); W=wald(b,V,R,np.zeros(1))
print('MRW n=',n,' sum=',s.item(),' se=',se.item(),' t=',(s/se).item(),' Wald=',W,' p=',stats.chi2.sf(W,1))

print('===== 9.24 Monte Carlo')
rng=np.random.default_rng(20260916)
B,n=1000,50; beta=1.0; theta=np.exp(beta)
bh=np.empty(B); th=np.empty(B); Tb=np.empty(B); Tt=np.empty(B)
for r in range(B):
    x=rng.uniform(size=n); e=rng.standard_normal(n); y=0.0+beta*x+e
    Xm=np.column_stack([np.ones(n),x]); b,V,_,_=ols(y,Xm,'HC1')
    sb=np.sqrt(V[1,1]); bh[r]=b[1]; th[r]=np.exp(b[1]); st=th[r]*sb
    Tb[r]=(b[1]-beta)/sb; Tt[r]=(th[r]-theta)/st
def mcse(p): return np.sqrt(p*(1-p)/B)
print(' E[bhat]=',bh.mean(),' mcse',bh.std(ddof=1)/np.sqrt(B))
print(' E[thetahat]=',th.mean(),' theta=',theta,' mcse',th.std(ddof=1)/np.sqrt(B))
print(' P[Tb>1.645]=',(Tb>1.645).mean(),' mcse',mcse((Tb>1.645).mean()))
print(' P[Tt>1.645]=',(Tt>1.645).mean(),' mcse',mcse((Tt>1.645).mean()))
print(' P[Tb<-1.645]=',(Tb<-1.645).mean(),' P[Tt<-1.645]=',(Tt<-1.645).mean())
# large-B check of the bias of exp(bhat)
rng2=np.random.default_rng(1); B2=20000; th2=np.empty(B2); tb2=np.empty(B2); tt2=np.empty(B2)
for r in range(B2):
    x=rng2.uniform(size=n); e=rng2.standard_normal(n); y=x+e
    Xm=np.column_stack([np.ones(n),x]); b,V,_,_=ols(y,Xm,'HC1'); sb=np.sqrt(V[1,1])
    th2[r]=np.exp(b[1]); tb2[r]=(b[1]-1)/sb; tt2[r]=(th2[r]-theta)/(th2[r]*sb)
print(' [B=20000] E[thetahat]=',th2.mean(),' P[Tb>1.645]=',(tb2>1.645).mean(),' P[Tt>1.645]=',(tt2>1.645).mean())
