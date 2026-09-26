import numpy as np, pandas as pd
from ivlib import *
from scipy import stats
from boot import bc
d=pd.read_stata("Card1995/Card1995.dta").astype({c:float for c in ["ed76","age76","black","reg76r","smsa76r","nearc2","nearc4","nearc4a","nearc4b","lwage76"]}); d=d[d.lwage76.notna()].copy(); n=len(d); one=np.ones(n)
print('n=',n)
y=d.lwage76.values; ed=d.ed76.values; age=d.age76.values; ex=age-ed-6
exo=np.column_stack([ex,ex**2/100,d.black,d.reg76r,d.smsa76r,one])  # exogenous (exp treated exogenous in 2SLS(a))
exl=['experience','exp2/100','black','south','urban','const']
pub=d.nearc4a.values; pri=d.nearc4b.values; col=d.nearc4.values
# (a) reduced form last column Table 12.2
Zr=np.column_stack([exo,pub,pri]); g,Vr,_=ols(ed,Zr); _,Vh,_=ols(ed,Zr,False)
for i,l in enumerate(exl+['public','private']): print(f'  RF {l:10s} {g[i]: .3f} ({np.sqrt(Vr[i,i]):.3f}) [{np.sqrt(Vh[i,i]):.3f}]')
Fh,q=first_stage_F(ed,exo,Zr,False); Fr,_=first_stage_F(ed,exo,Zr,True); print('  F homo',Fh,' F robust',Fr)
X=np.column_stack([ed,exo]); b,Vr,e=tsls(y,X,Zr); _,Vh,_=tsls(y,X,Zr,False)
for i,l in enumerate(['education']+exl): print(f'  2SLS(a) {l:10s} {b[i]: .3f} ({np.sqrt(Vr[i,i]):.3f}) [{np.sqrt(Vh[i,i]):.3f}]')
print('  Sargan',sargan(e,Zr))
# (b) add nearc2
Zb=np.column_stack([exo,pub,pri,d.nearc2]); gb,Vb,_=ols(ed,Zb)
print('  (b) nearc2 coef',gb[-1],' se',np.sqrt(Vb[-1,-1]),' t',gb[-1]/np.sqrt(Vb[-1,-1]))
Fb,_=first_stage_F(ed,exo,Zb,False); print('      F homo with nearc2',Fb)
bb,Vbb,_=tsls(y,X,Zb); print('      2SLS with nearc2 educ',bb[0],np.sqrt(Vbb[0,0]))
# (c) interactions
ia=pub*age; ia2=pub*age**2/100
Zc=np.column_stack([exo,pub,pri,ia,ia2]); gc,Vc,_=ols(ed,Zc); _,Vch,_=ols(ed,Zc,False)
for i,l in zip(range(6,10),['public','private','pub*age','pub*age2/100']): print(f'  (c) {l:12s} {gc[i]: .4f} ({np.sqrt(Vc[i,i]):.4f})')
Rj=np.zeros((10,2)); Rj[8,0]=1; Rj[9,1]=1
dj=Rj.T@gc; print('      joint Wald interactions',dj@np.linalg.solve(Rj.T@Vc@Rj,dj))
agegrid=np.array([24,26,28,30,32,34]); print('      effect of public by age:',[round(gc[6]+gc[8]*a+gc[9]*a*a/100,3) for a in agegrid])
print('      age range',age.min(),age.max())
# (d) 2SLS with expanded set
Zd=Zc; bd,Vdr,ed_=tsls(y,X,Zd); _,Vdh,_=tsls(y,X,Zd,False)
print('  (d) 2SLS educ',bd[0],' se rob',np.sqrt(Vdr[0,0]),' homo',np.sqrt(Vdh[0,0]))
for i,l in enumerate(['education']+exl): print(f'      {l:10s} {bd[i]: .3f} ({np.sqrt(Vdr[i,i]):.3f})')
Fd,q=first_stage_F(ed,exo,Zd,False); Fdr,_=first_stage_F(ed,exo,Zd,True); print('  (e) F homo',Fd,' robust',Fdr,' q',q)
print('      Sargan',sargan(ed_,Zd),' p',stats.chi2.sf(sargan(ed_,Zd),3))
# (f) exogeneity test (control function)
_,_,uh=ols(ed,Zd,False); bf,Vf,_=ols(y,np.column_stack([X,uh])); _,Vfh,_=ols(y,np.column_stack([X,uh]),False)
print('  (f) CF uhat coef',bf[-1],' t rob',bf[-1]/np.sqrt(Vf[-1,-1]),' t homo',bf[-1]/np.sqrt(Vfh[-1,-1]))
# (g) LIML
bl,Vl,el,kap=liml(y,ed[:,None],exo,Zd); _,Vlh,_,_=liml(y,ed[:,None],exo,Zd,False)
print('  (g) LIML educ',bl[0],' se rob',np.sqrt(Vl[0,0]),' homo',np.sqrt(Vlh[0,0]),' kappa',kap)
for i,l in enumerate(['education']+exl): print(f'      {l:10s} {bl[i]: .3f} ({np.sqrt(Vl[i,i]):.3f})')
# check LIML for 2SLS(a)
bla,Vla,_,ka=liml(y,ed[:,None],exo,Zr); print('  LIML(a) check educ',bla[0],np.sqrt(Vla[0,0]))
# 12.25 IV(a): college instrument
Za=np.column_stack([exo,col]); bi,Vi,_=tsls(y,X,Za); _,Vih,_=tsls(y,X,Za,False)
print('  IV(a) educ',bi[0],' rob',np.sqrt(Vi[0,0]),' homo',np.sqrt(Vih[0,0]))
rng=np.random.default_rng(1225); B=10000; bs=np.empty(B)
for k in range(B):
    idx=rng.integers(0,n,n); bs[k]=tsls(y[idx],X[idx],Za[idx],robust=False)[0][0]
ci,z0=bc(bs,bi[0]); print('  12.25 boot se',bs.std(ddof=1),' BC',ci,' z0',z0,' pct',np.quantile(bs,[.025,.975]),' min/max',bs.min(),bs.max())
print('         IQR-based sd',(np.quantile(bs,.75)-np.quantile(bs,.25))/1.349)
