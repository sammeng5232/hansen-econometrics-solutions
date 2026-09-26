import numpy as np, pandas as pd
from scipy import stats
def reg(y,X,hac=None):
    n,k=X.shape; XXi=np.linalg.inv(X.T@X); b=XXi@(X.T@y); e=y-X@b
    if hac is None:
        S=(X*(e**2)[:,None]).T@X; V=(n/(n-k))*XXi@S@XXi
    else:
        u=X*e[:,None]; S=u.T@u
        for l in range(1,hac+1):
            w=1-l/(hac+1); G=u[l:].T@u[:-l]; S+=w*(G+G.T)
        V=(n/(n-k))*XXi@S@XXi
    return b,V,e
def lagmat(x,L): return np.column_stack([np.r_[np.full(l,np.nan),x[:-l]] for l in L])
def wald(b,V,idx):
    d=b[idx]; return float(d@np.linalg.solve(V[np.ix_(idx,idx)],d))
q=pd.read_stata('progs/FRED-QD.dta'); m=pd.read_stata('progs/FRED-MD.dta')
def fit(y,Xs,names,hac=None,label=''):
    D=np.column_stack([y]+Xs); ok=~np.isnan(D).any(1); D=D[ok]; n=len(D)
    X=np.column_stack([np.ones(n),D[:,1:]]); b,V,e=reg(D[:,0],X,hac)
    print(f'  {label} n={n}')
    for i,nm in enumerate(['const']+names): print(f'     {nm:10s} {b[i]: .4f} ({np.sqrt(V[i,i]):.4f})')
    return b,V,e,n
print('=== 14.18 pnfix')
x=q.pnfix.values.astype(float); g=100*(x[1:]/x[:-1]-1); g=np.r_[np.nan,g]
b,V,e,n=fit(g,[lagmat(g,[1,2,3,4])],['L1','L2','L3','L4'],None,'HC1')
b2,V2,_,_=fit(g,[lagmat(g,[1,2,3,4])],['L1','L2','L3','L4'],5,'Newey-West M=5')
a=b[1:5]; print('  sum AR',a.sum(),' mean growth',np.nanmean(g), ' implied mean',b[0]/(1-a.sum()))
bj=[1.0]; hist=[1.0,0,0,0]
for j in range(1,11):
    val=sum(a[i]*(bj[j-1-i] if j-1-i>=0 else 0) for i in range(4)); bj.append(val)
print('  IRF',np.round(bj,4))
roots=np.roots(np.r_[1,-a]); print('  |roots of z^4 - a1 z^3...|',np.round(np.abs(roots),3))
print('=== 14.19 oilpricex')
o=q.oilpricex.values.astype(float); do=np.r_[np.nan,np.diff(o)]
b,V,e,n=fit(do,[lagmat(do,[1,2,3,4])],['L1','L2','L3','L4'],None,'HC1')
W=wald(b,V,[1,2,3,4]); print('  Wald AR=0',W,' p',stats.chi2.sf(W,4))
b2,V2,_,_=fit(do,[lagmat(do,[1,2,3,4])],['L1','L2','L3','L4'],5,'NW'); W2=wald(b2,V2,[1,2,3,4]); print('  NW Wald',W2,stats.chi2.sf(W2,4))
print('  sd of diff',np.nanstd(do),' R2', 1-(e@e)/((do[~np.isnan(do)][-n:]-do[~np.isnan(do)][-n:].mean())**2).sum())
print('=== 14.20 unrate AIC')
u=m.unrate.values.astype(float); tm=m.time.values
start=np.where(tm==np.datetime64('1960-01-01'))[0][0]; y=u[start:]; n=len(y)
for p in range(1,9):
    X=np.column_stack([np.ones(n)]+[u[start-l:len(u)-l] for l in range(1,p+1)])
    bb=np.linalg.lstsq(X,y,rcond=None)[0]; ee=y-X@bb; s2=ee@ee/n
    print(f'  AR({p}) n={n} sigma2={s2:.6f} AIC={n*np.log(s2)+2*p:.2f}')
p=5
X=np.column_stack([np.ones(n)]+[u[start-l:len(u)-l] for l in range(1,9)])
for p in [4,5,6,8]:
    Xp=X[:,:p+1]; bb,VV,ee=reg(y,Xp); print(f'  AR({p}) coefs',np.round(bb,4),' se',np.round(np.sqrt(np.diag(VV)),4),' sum',round(bb[1:].sum(),4))
print('=== 14.21 unrate on claims')
ur=q.unrate.values.astype(float); cl=q.claimsx.values.astype(float)/1000
b,V,e,n=fit(ur,[lagmat(cl,[1,2,3,4])],['cl1','cl2','cl3','cl4'],None,'DL HC1')
b,V,e,n=fit(ur,[lagmat(cl,[1,2,3,4])],['cl1','cl2','cl3','cl4'],5,'DL NW5')
r1=np.corrcoef(e[1:],e[:-1])[0,1]; print('  DL residual AR1 corr',r1)
b,V,e,n=fit(ur,[lagmat(ur,[1,2,3,4]),lagmat(cl,[1,2,3,4])],['ur1','ur2','ur3','ur4','cl1','cl2','cl3','cl4'],None,'ADL HC1')
W=wald(b,V,[5,6,7,8]); print('  Granger Wald',W,' p',stats.chi2.sf(W,4),' sum cl',b[5:9].sum(),' sum ur',b[1:5].sum())
print('  ADL residual AR1 corr',np.corrcoef(e[1:],e[:-1])[0,1])
print('=== 14.22 gdp on houst')
gd=q.gdpc1.values.astype(float); gg=np.r_[np.nan,100*(gd[1:]/gd[:-1]-1)]; h=q.houst.values.astype(float)/1000
b,V,e,n=fit(gg,[lagmat(h,[1,2,3,4])],['h1','h2','h3','h4'],None,'DL HC1')
b,V,e,n=fit(gg,[lagmat(h,[1,2,3,4])],['h1','h2','h3','h4'],5,'DL NW5')
print('  DL residual AR1 corr',np.corrcoef(e[1:],e[:-1])[0,1], ' sum h',b[1:].sum())
b,V,e,n=fit(gg,[lagmat(gg,[1,2]),lagmat(h,[1,2,3,4])],['g1','g2','h1','h2','h3','h4'],None,'ADL HC1')
W=wald(b,V,[3,4,5,6]); print('  Granger Wald',W,' p',stats.chi2.sf(W,4),' sum h',b[3:7].sum())
print('  houst mean',np.nanmean(h),' sd',np.nanstd(h))
