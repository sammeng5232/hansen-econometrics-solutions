import numpy as np, pandas as pd
from boot import *
df=pd.read_stata('DDK2011/DDK2011.dta')
cols=['tracking','agetest','girl','etpteacher','percentile']
D=df[['totalscore','schoolid']+cols].dropna()
y=((D.totalscore-D.totalscore.mean())/D.totalscore.std(ddof=1)).values
X=np.column_stack([D[c].values.astype(float) for c in cols]+[np.ones(len(D))]); g=D.schoolid.values
n,k=X.shape; b=fit(y,X); e=y-X@b; XXi=np.linalg.inv(X.T@X)
S=np.zeros((k,k)); ug=np.unique(g); G=len(ug)
for gg in ug:
    s=(X[g==gg]*e[g==gg][:,None]).sum(0); S+=np.outer(s,s)
Vc=((n-1)/(n-k))*(G/(G-1))*XXi@S@XXi
jb,Vj=jack(y,X,fit,groups=g)
rng=np.random.default_rng(1031); B=5000
bd=bootdraws(y,X,fit,B,rng,groups=g); Vb=np.cov(bd.T)
print('n',n,'G',G)
for j,t in enumerate(cols+['intercept']):
    ci,z0,ah=bca(bd[:,j],b[j],jb[:,j])
    print(f'  {t:11s} {b[j]: .4f}  cluster-asy {np.sqrt(Vc[j,j]):.4f}  cl-jack {np.sqrt(Vj[j,j]):.4f}  cl-boot {np.sqrt(Vb[j,j]):.4f}  BCa [{ci[0]:.4f},{ci[1]:.4f}]  z0={z0:.3f} a={ah:.4f}')
