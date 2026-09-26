import os
import numpy as np, pandas as pd
from lib import ols, wald
from scipy import stats
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 't42.py')).read().split("m=edu>=12")[0])
print("===== 9.28 non-Hispanic Black, educ>=12")
m=(edu>=12)&(race==2)&(hisp==0)
X,nm=design(m,drop=('hispanic','black','amerind','asian','mixed')); b,V,e,_=ols(y[m],X,'HC2')
print(' n=',m.sum(),' k=',X.shape[1])
for j,t in enumerate(nm): print(f'  {t:14s} {b[j]: .4f} ({np.sqrt(V[j,j]):.4f})')
idx=[nm.index(t) for t in ['married_fem','married_male','formerly_fem','formerly_male']]
R=np.zeros((X.shape[1],4)); 
for c,i in enumerate(idx): R[i,c]=1
W=wald(b,V,R,np.zeros(4)); print(' Wald=',round(W,3),' p(chi2_4)=',stats.chi2.sf(W,4),' F=W/4=',round(W/4,3),' p(F)=',stats.f.sf(W/4,4,X.shape[0]-X.shape[1]))

print("===== 9.29 non-Hispanic white & Black, educ>=12")
m=(edu>=12)&np.isin(race,[1,2])&(hisp==0)
fem=female[m]; blk=(race[m]==2)*1.0; ed=edu[m]; xp=ex[m]; un=union[m]; mr=marital[m]
groups={'wm':(1-blk)*(1-fem),'wf':(1-blk)*fem,'bm':blk*(1-fem),'bf':blk*fem}
cols=[ed*groups[g] for g in ['wm','wf','bm','bf']]+[xp,xp**2/100,fem,blk,blk*fem,fem*un,(1-fem)*un,
      fem*np.isin(mr,[1,2,3]),(1-fem)*np.isin(mr,[1,2,3]),fem*np.isin(mr,[4,5,6]),(1-fem)*np.isin(mr,[4,5,6]),np.ones(int(m.sum()))]
nm=['educ_wm','educ_wf','educ_bm','educ_bf','experience','exp2/100','female','black','black*female','fem_union','male_union','married_fem','married_male','formerly_fem','formerly_male','intercept']
X=np.column_stack(cols); b,V,e,_=ols(y[m],X,'HC2')
print(' n=',m.sum(),' groups:',{g:int(v.sum()) for g,v in groups.items()})
for j,t in enumerate(nm): print(f'  {t:14s} {b[j]: .4f} ({np.sqrt(V[j,j]):.4f})')
R=np.zeros((X.shape[1],3)); R[0,:]=1; R[1,0]=-1; R[2,1]=-1; R[3,2]=-1
W=wald(b,V,R,np.zeros(3)); print(' Wald=',round(W,3),' p(chi2_3)=',stats.chi2.sf(W,3),' F=',round(W/3,3))

print("===== 9.25 Invest1993")
dd=pd.read_stata('Invest1993/Invest1993.dta'); dd=dd[dd.year==1987]
print(' n=',len(dd))
X=np.column_stack([dd.vala,dd.cfa,dd.debta,np.ones(len(dd))]); yI=dd.inva.values
nm=['Q','C','D','intercept']
for hc in ['HC1']:
    b,V,e,_=ols(yI,X,hc)
    for j,t in enumerate(nm):
        s=np.sqrt(V[j,j]); print(f'  {t:9s} {b[j]: .5f} ({s:.5f})  95%CI [{b[j]-1.96*s:.5f},{b[j]+1.96*s:.5f}] t={b[j]/s:.2f}')
R=np.zeros((4,2)); R[1,0]=1; R[2,1]=1
W=wald(b,V,R,np.zeros(2)); print('  Wald C=D=0:',round(W,3),'p=',stats.chi2.sf(W,2))
q,c_,dv=dd.vala.values,dd.cfa.values,dd.debta.values
X2=np.column_stack([q,c_,dv,q**2,c_**2,dv**2,q*c_,q*dv,c_*dv,np.ones(len(dd))])
b2,V2,_,_=ols(yI,X2,'HC1')
R=np.zeros((10,6))
for c,i in enumerate(range(3,9)): R[i,c]=1
W=wald(b2,V2,R,np.zeros(6)); print('  quadratic: Wald 6 restr =',round(W,3),'p=',stats.chi2.sf(W,6))
nm2=['Q','C','D','Q2','C2','D2','QC','QD','CD','int']
for j,t in enumerate(nm2): print(f'    {t:4s} {b2[j]: .5f} ({np.sqrt(V2[j,j]):.5f})')

print("===== 9.26 Nerlove")
nd=pd.read_stata('Nerlove1963/Nerlove1963.dta'); n=len(nd)
yN=np.log(nd.cost.values)
X=np.column_stack([np.ones(n),np.log(nd.output),np.log(nd.Plabor),np.log(nd.Pcapital),np.log(nd.Pfuel)])
nm=['const','logQ','logPL','logPK','logPF']; k=5
b,V,e,XXi=ols(yN,X,'HC1')
for j,t in enumerate(nm): print(f'  OLS {t:6s} {b[j]: .4f} ({np.sqrt(V[j,j]):.4f})')
R=np.array([[0,0,1,1,1.]]).T; cc=np.array([1.])
A=XXi@R@np.linalg.inv(R.T@XXi@R); bt=b-A@(R.T@b-cc); et=yN-X@bt
Om=(n/(n-k+1))*(X.T@(X*(et**2)[:,None])); H=np.eye(k)-A@R.T; Vc=H@(XXi@Om@XXi)@H.T
for j,t in enumerate(nm): print(f'  CLS {t:6s} {bt[j]: .4f} ({np.sqrt(Vc[j,j]):.4f})')
bm=b-V@R@np.linalg.inv(R.T@V@R)@(R.T@b-cc); Vm=V-V@R@np.linalg.inv(R.T@V@R)@R.T@V
for j,t in enumerate(nm): print(f'  EMD {t:6s} {bm[j]: .4f} ({np.sqrt(Vm[j,j]):.4f})')
W=wald(b,V,R,cc); print('  Wald =',round(W,4),'p=',stats.chi2.sf(W,1),' sum=',R.T@b)
J=float((b-bm)@np.linalg.solve(V,(b-bm))); print('  MD J =',round(J,4),'p=',stats.chi2.sf(J,1))
# homoskedastic-weight MD (CLS criterion) for comparison
s2=(e@e)/(n-k); J0=float((b-bt)@(X.T@X)@(b-bt))/s2; print('  J0 (CLS criterion/s2)=',round(J0,4))

print("===== 9.27 MRW")
md=pd.read_stata('MRW1992/MRW1992.dta'); md=md[md.N==1]; n=len(md)
yM=np.log(md.Y85.values)-np.log(md.Y60.values)
X=np.column_stack([np.log(md.Y60),np.log(md.invest/100),np.log(md.pop_growth/100+0.05),np.log(md.school/100),np.ones(n)])
b,V,e,_=ols(yM,X,'HC1')
for j,t in enumerate(['lnY60','lnI','lnG','lnS','int']): print(f'  {t:6s} {b[j]: .4f} ({np.sqrt(V[j,j]):.4f})')
R=np.array([[0,1,1,1,0.]]).T
W=wald(b,V,R,np.zeros(1)); print('  n=',n,' sum=',(R.T@b).item(),' se=',np.sqrt(R.T@V@R).item(),' Wald=',round(W,4),'p=',stats.chi2.sf(W,1))
