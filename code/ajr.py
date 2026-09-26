import numpy as np, pandas as pd
from ivlib import *
from scipy import stats
d=pd.read_stata('AJR2001/AJR2001.dta'); n=len(d); one=np.ones(n)
y=d.loggdp.values; r=d.risk.values; lm=d.logmort0.values
def show(name,b,Vr,Vh,idx,labels):
    print(' ',name)
    for i,l in zip(idx,labels): print(f'     {l:10s} {b[i]: .4f}  homo {np.sqrt(Vh[i,i]):.4f}  robust {np.sqrt(Vr[i,i]):.4f}')
b,Vr,_=ols(y,np.column_stack([r,one])); _,Vh,_=ols(y,np.column_stack([r,one]),False); show('(12.86) OLS',b,Vr,Vh,[0,1],['risk','const'])
g,Vr,u=ols(r,np.column_stack([lm,one])); _,Vh,_=ols(r,np.column_stack([lm,one]),False); show('(12.87) first stage',g,Vr,Vh,[0,1],['logmort','const'])
print('     t(homo)',g[0]/np.sqrt(Vh[0,0]),' F(homo)',(g[0]/np.sqrt(Vh[0,0]))**2,' t(rob)',g[0]/np.sqrt(Vr[0,0]))
X=np.column_stack([r,one]); Z=np.column_stack([lm,one])
b2,Vr,e2=tsls(y,X,Z); _,Vh,_=tsls(y,X,Z,False); show('(12.88) 2SLS',b2,Vr,Vh,[0,1],['risk','const'])
lam,_,_=ols(y,Z); print('  (c) ILS lambda/gamma =',lam[0]/g[0],' lambda=',lam[0])
rh=Z@g; b2s,_,_=ols(y,np.column_stack([rh,one])); print('  (d) two-stage coef =',b2s[0])
bc,Vcr,_=ols(y,np.column_stack([r,u,one])); _,Vch,_=ols(y,np.column_stack([r,u,one]),False)
print('  (e) control fn: risk',bc[0],' uhat',bc[1],' t_u homo',bc[1]/np.sqrt(Vch[1,1]),' robust',bc[1]/np.sqrt(Vcr[1,1]))
lat=d.latitude.values; af=d.africa.values
Xf=np.column_stack([r,lat,af,one]); b,Vr,_=ols(y,Xf); _,Vh,_=ols(y,Xf,False); show('(f) OLS +lat,africa',b,Vr,Vh,[0,1,2],['risk','latitude','africa'])
Zf=np.column_stack([lm,lat,af,one]); b,Vr,_=tsls(y,Xf,Zf); _,Vh,_=tsls(y,Xf,Zf,False); show('(g) 2SLS +lat,africa',b,Vr,Vh,[0,1,2],['risk','latitude','africa'])
gg,Vr,_=ols(r,Zf,False); print('     first-stage t logmort (g):',gg[0]/np.sqrt(Vr[0,0]))
mort=np.exp(lm)
for nm,inst in [('logmort',lm),('mortality',mort)]:
    gm,Vr,_=ols(r,np.column_stack([inst,one])); _,Vh,uu=ols(r,np.column_stack([inst,one]),False)
    R2=1-(uu@uu)/((r-r.mean())@(r-r.mean()))
    print(f'  (h) first stage on {nm}: coef {gm[0]:.5f} se_h {np.sqrt(Vh[0,0]):.5f} se_r {np.sqrt(Vr[0,0]):.5f} t_h {gm[0]/np.sqrt(Vh[0,0]):.2f} R2 {R2:.3f}')
print('  mortality summary: min',mort.min(),' median',np.median(mort),' max',mort.max())
Zi=np.column_stack([lm,lm**2,one])
gi,Vr,_=ols(r,Zi); _,Vh,_=ols(r,Zi,False)
for i,l in enumerate(['logmort','logmort^2','const']): print(f'  (i) FS {l:10s} {gi[i]: .4f} homo {np.sqrt(Vh[i,i]):.4f} rob {np.sqrt(Vr[i,i]):.4f}')
Fh,q=first_stage_F(r,one[:,None],Zi,False); Fr,_=first_stage_F(r,one[:,None],Zi,True)
print('  (j) F homo',Fh,' F robust',Fr,' q',q)
bi,Vr,ei=tsls(y,X,Zi); _,Vh,_=tsls(y,X,Zi,False); show('(i) 2SLS logmort, logmort^2',bi,Vr,Vh,[0,1],['risk','const'])
S=sargan(ei,Zi); print('  (k) Sargan',S,' p',stats.chi2.sf(S,1))
bl,Vlr,el,kap=liml(y,r[:,None],one[:,None],Zi); _,Vlh,_,_=liml(y,r[:,None],one[:,None],Zi,False)
print('  (l) LIML risk',bl[0],' se homo',np.sqrt(Vlh[0,0]),' robust',np.sqrt(Vlr[0,0]),' kappa',kap)
# bootstrap 12.23
for seed in [1,2]:
    rng=np.random.default_rng(seed); B=10000; bs=np.empty(B)
    for bb in range(B):
        idx=rng.integers(0,n,n)
        bs[bb]=tsls(y[idx],X[idx],Z[idx],robust=False)[0][0]
    print(f'  12.23 seed {seed}: boot se {bs.std(ddof=1):.4f}  IQR/1.349 {(np.quantile(bs,.75)-np.quantile(bs,.25))/1.349:.4f} min {bs.min():.2f} max {bs.max():.2f} q01 {np.quantile(bs,.01):.3f} q99 {np.quantile(bs,.99):.3f}')
