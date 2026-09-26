import numpy as np
np.set_printoptions(suppress=True, precision=6)
d=np.loadtxt('cps09mar/cps09mar.txt')
age,female,hisp,edu,earn,hours,week,union,uncov,region,race,marital=[d[:,i] for i in range(12)]
ex=age-edu-6
sam=(race==1)&(female==0)&(hisp==1); n=int(sam.sum())
y=np.log(earn[sam]/(hours[sam]*week[sam]))
ed=edu[sam]; xp=ex[sam]; mr=marital[sam]
cols=[ed,xp,xp**2/100]+[(mr==c)*1.0 for c in [1,2,3,4,5,6]]+[np.ones(n)]
X=np.column_stack(cols); k=X.shape[1]
names=['education','experience','exp^2/100','married1','married2','married3',
       'widowed','divorced','separated','intercept']
print('n=',n,'k=',k,'counts married1..sep:',[int((mr==c).sum()) for c in [1,2,3,4,5,6,7]])
XXi=np.linalg.inv(X.T@X); b=XXi@(X.T@y); e=y-X@b
S=(n/(n-k))*XXi@(X.T@(X*(e**2)[:,None]))@XXi           # HC1 estimate of var(bhat)
print('\n(a) OLS')
for j,nm in enumerate(names): print(f'   {nm:11s} {b[j]: .6f}  ({np.sqrt(S[j,j]):.6f})')

# constraints: b4=b7 (married1=widowed), b8=b9 (divorced=separated)
R=np.zeros((k,2)); R[3,0]=1; R[6,0]=-1; R[7,1]=1; R[8,1]=-1; c=np.zeros(2); q=2
# ---- CLS
A=XXi@R@np.linalg.inv(R.T@XXi@R)
bt=b-A@(R.T@b-c); et=y-X@bt
Om=(n/(n-k+q))*(X.T@(X*(et**2)[:,None]))
Sc=XXi@Om@XXi
H=np.eye(k)-A@R.T
Vcls=H@Sc@H.T
print('\n(b) CLS  (beta4=beta7, beta8=beta9)')
for j,nm in enumerate(names): print(f'   {nm:11s} {bt[j]: .6f}  ({np.sqrt(Vcls[j,j]):.6f})')
print('   SSE=',round(et@et,4),' check R\'b:',np.round(R.T@bt,12))
# ---- EMD
bm=b-S@R@np.linalg.inv(R.T@S@R)@(R.T@b-c)
Vemd=S-S@R@np.linalg.inv(R.T@S@R)@R.T@S
em=y-X@bm
print('\n(c) EMD  (same constraints)')
for j,nm in enumerate(names): print(f'   {nm:11s} {bm[j]: .6f}  ({np.sqrt(Vemd[j,j]):.6f})')
print('   SSE=',round(em@em,4))
# ---- (d)/(e)
print('\n(d) b2+b3 at OLS =',round(b[1]+b[2],6),'  at CLS =',round(bt[1]+bt[2],6))
R2=np.zeros((k,3)); R2[:,:2]=R; R2[1,2]=1; R2[2,2]=1; c2=np.zeros(3)
A2=XXi@R2@np.linalg.inv(R2.T@XXi@R2)
be=b-A2@(R2.T@b-c2); ee=y-X@be
Om2=(n/(n-k+3))*(X.T@(X*(ee**2)[:,None]))
H2=np.eye(k)-A2@R2.T; V2=H2@(XXi@Om2@XXi)@H2.T
print('\n(e) CLS with beta4=beta7, beta8=beta9, beta2+beta3=0')
for j,nm in enumerate(names): print(f'   {nm:11s} {be[j]: .6f}  ({np.sqrt(V2[j,j]):.6f})')
print('   SSE=',round(ee@ee,4),' b2+b3 =',round(be[1]+be[2],10))
print('\n   SSE: OLS',round(e@e,4),' CLS(b)',round(et@et,4),' CLS(e)',round(ee@ee,4))
