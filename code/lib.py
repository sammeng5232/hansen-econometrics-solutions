import numpy as np
def ols(y,X,hc='HC1'):
    n,k=X.shape
    XXi=np.linalg.inv(X.T@X); b=XXi@(X.T@y); e=y-X@b
    h=np.einsum('ij,jk,ik->i',X,XXi,X)
    if hc=='HC0': w=e**2
    elif hc=='HC1': w=e**2*n/(n-k)
    elif hc=='HC2': w=e**2/(1-h)
    elif hc=='HC3': w=e**2/(1-h)**2
    V=XXi@(X.T@(X*w[:,None]))@XXi
    return b,V,e,XXi
def wald(b,V,R,c):
    d=R.T@b-c
    return float(d@np.linalg.solve(R.T@V@R,d))
