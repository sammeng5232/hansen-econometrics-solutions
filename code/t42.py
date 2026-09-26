import numpy as np
from lib import ols
from scipy import stats
d=np.loadtxt('cps09mar/cps09mar.txt')
age,female,hisp,edu,earn,hours,week,union,uncov,region,race,marital=[d[:,i] for i in range(12)]
ex=age-edu-6; y=np.log(earn/(hours*week))
def design(m, drop=()):
    fem=female[m]; un=union[m]; mr=marital[m]; rc=race[m]
    cols={'education':edu[m],'experience':ex[m],'exp2/100':ex[m]**2/100,'female':fem,
      'fem_union':fem*un,'male_union':(1-fem)*un,'married_fem':fem*np.isin(mr,[1,2,3]),
      'married_male':(1-fem)*np.isin(mr,[1,2,3]),'formerly_fem':fem*np.isin(mr,[4,5,6]),
      'formerly_male':(1-fem)*np.isin(mr,[4,5,6]),'hispanic':hisp[m],'black':(rc==2)*1.0,
      'amerind':(rc==3)*1.0,'asian':(rc==4)*1.0,'mixed':(rc>=6)*1.0}
    names=[k for k in cols if k not in drop]
    X=np.column_stack([cols[k] for k in names]+[np.ones(int(m.sum()))])
    return X,names+['intercept']
m=edu>=12
X,nm=design(m); b,V,e,_=ols(y[m],X,'HC2')
print('Table 4.2 replication n=',m.sum())
for j,t in enumerate(nm): print(f'  {t:14s} {b[j]: .3f} ({np.sqrt(V[j,j]):.3f})')
