import numpy as np
from scipy import stats
from lib import ols
Phi=stats.norm.cdf; iPhi=stats.norm.ppf
def fit(y,X): return np.linalg.solve(X.T@X,X.T@y)
def jack(y,X,fn,groups=None):
    if groups is None:
        n=len(y); out=np.array([fn(np.delete(y,i),np.delete(X,i,0)) for i in range(n)])
    else:
        ug=np.unique(groups); out=np.array([fn(y[groups!=g],X[groups!=g]) for g in ug]); n=len(ug)
    m=out.mean(0); V=(n-1)/n*((out-m).T@(out-m)) if out.ndim>1 else (n-1)/n*((out-m)**2).sum()
    return out,V
def bootdraws(y,X,fn,B,rng,groups=None):
    n=len(y); res=[]
    if groups is None:
        for b in range(B):
            idx=rng.integers(0,n,n); res.append(fn(y[idx],X[idx]))
    else:
        ug=np.unique(groups); pos={g:np.where(groups==g)[0] for g in ug}; G=len(ug)
        for b in range(B):
            gs=ug[rng.integers(0,G,G)]; idx=np.concatenate([pos[g] for g in gs])
            res.append(fn(y[idx],X[idx]))
    return np.array(res)
def pct(d,a=0.05): return np.quantile(d,[a/2,1-a/2])
def bc(d,th,a=0.05):
    z0=iPhi((d<=th).mean()); x=Phi(iPhi([a/2,1-a/2])+2*z0); return np.quantile(d,x),z0
def bca(d,th,jk,a=0.05):
    z0=iPhi((d<=th).mean()); m=jk.mean(); ah=((m-jk)**3).sum()/(6*(((m-jk)**2).sum())**1.5)
    za=iPhi([a/2,1-a/2]); x=Phi(z0+(za+z0)/(1-ah*(za+z0))); return np.quantile(d,x),z0,ah
