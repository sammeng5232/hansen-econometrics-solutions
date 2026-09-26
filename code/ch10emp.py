import numpy as np, pandas as pd
from lib import ols
from boot import *
B=10000
print("===== 10.28 Nerlove")
nd=pd.read_stata('Nerlove1963/Nerlove1963.dta'); n=len(nd)
y=np.log(nd.cost.values)
X=np.column_stack([np.ones(n),np.log(nd.output),np.log(nd.Plabor),np.log(nd.Pcapital),np.log(nd.Pfuel)])
b,V,_,_=ols(y,X,'HC1'); rng=np.random.default_rng(1028)
jb,Vj=jack(y,X,fit); bd=bootdraws(y,X,fit,B,rng); Vb=np.cov(bd.T)
for j,t in enumerate(['const','logQ','logPL','logPK','logPF']):
    print(f'  {t:6s} {b[j]: .4f}  asy {np.sqrt(V[j,j]):.4f}  jack {np.sqrt(Vj[j,j]):.4f}  boot {np.sqrt(Vb[j,j]):.4f}')
R=np.array([0,0,1,1,1.]); th=R@b; thj=jb@R; thb=bd@R
print('  theta',th,' asy',np.sqrt(R@V@R),' jack',np.sqrt(Vj[2:,2:].sum()),' boot',thb.std(ddof=1))
print('  pct',pct(thb),' BCa',bca(thb,th,thj),' BC',bc(thb,th))
print("===== 10.29 MRW")
md=pd.read_stata('MRW1992/MRW1992.dta'); md=md[md.N==1]; n=len(md)
y=np.log(md.Y85.values)-np.log(md.Y60.values)
X=np.column_stack([np.log(md.Y60),np.log(md.invest/100),np.log(md.pop_growth/100+0.05),np.log(md.school/100),np.ones(n)])
b,V,_,_=ols(y,X,'HC1'); rng=np.random.default_rng(1029)
jb,Vj=jack(y,X,fit); bd=bootdraws(y,X,fit,B,rng); Vb=np.cov(bd.T)
for j,t in enumerate(['lnY60','lnI','lnG','lnS','const']):
    print(f'  {t:6s} {b[j]: .4f}  asy {np.sqrt(V[j,j]):.4f}  jack {np.sqrt(Vj[j,j]):.4f}  boot {np.sqrt(Vb[j,j]):.4f}')
R=np.array([0,1,1,1,0.]); th=R@b; thj=jb@R; thb=bd@R
print('  theta',th,' asy',np.sqrt(R@V@R),' jack',np.sqrt(R@Vj@R),' boot',thb.std(ddof=1))
print('  pct',pct(thb),' BC',bc(thb,th),' BCa',bca(thb,th,thj))
print("===== 10.30 CPS ratio")
d=np.loadtxt('cps09mar/cps09mar.txt')
age,female,hisp,edu,earn,hours,week,union,uncov,region,race,marital=[d[:,i] for i in range(12)]
ex=age-edu-6
m=(race==1)&(female==0)&(hisp==1)&(marital==7)&(region==2); n=int(m.sum())
y=np.log(earn[m]/(hours[m]*week[m])); X=np.column_stack([edu[m],ex[m],ex[m]**2/100,np.ones(n)])
def theta(b): return b[0]/(b[1]+0.2*b[2])
def fth(yy,XX): return theta(fit(yy,XX))
b,V,_,_=ols(y,X,'HC1'); th=theta(b); den=b[1]+0.2*b[2]
G=np.array([1/den,-b[0]/den**2,-0.2*b[0]/den**2,0])
print('  n=',n,' b=',b,' theta=',th,' denom=',den,' se(denom)=',np.sqrt(np.array([0,1,.2,0])@V@np.array([0,1,.2,0])))
thj,Vj=jack(y,X,fth); rng=np.random.default_rng(1030); thb=bootdraws(y,X,fth,B,rng)
print('  asy se',np.sqrt(G@V@G),' jack',np.sqrt(Vj),' boot',thb.std(ddof=1))
print('  boot quantiles 1,5,25,50,75,95,99:',np.quantile(thb,[.01,.05,.25,.5,.75,.95,.99]))
print('  share of draws with denom<=0:',np.mean(bootdraws(y,X,lambda yy,XX: (lambda bb: bb[1]+0.2*bb[2])(fit(yy,XX)),2000,np.random.default_rng(7))<=0))
print('  BC',bc(thb,th),' pct',pct(thb),' asy CI',th-1.96*np.sqrt(G@V@G),th+1.96*np.sqrt(G@V@G))
print('  jackknife theta range',thj.min(),thj.max())
