import numpy as np
from scipy import stats

def ols(y, X, robust=True):
    n, k = X.shape
    XXi = np.linalg.inv(X.T @ X); b = XXi @ (X.T @ y); e = y - X @ b
    if robust:
        V = (n / (n - k)) * XXi @ (X.T @ (X * (e ** 2)[:, None])) @ XXi
    else:
        V = (e @ e / (n - k)) * XXi
    return b, V, e

def tsls(y, X, Z, robust=True, kclass=None):
    """2SLS (kclass=None) or k-class with given kappa. X, Z include exogenous columns."""
    n, k = X.shape
    ZZi = np.linalg.pinv(Z.T @ Z)
    PzX = Z @ (ZZi @ (Z.T @ X))
    if kclass is None:
        A = X.T @ PzX
        bvec = X.T @ (Z @ (ZZi @ (Z.T @ y)))
    else:
        MzX = X - PzX
        Pzy = Z @ (ZZi @ (Z.T @ y))
        A = X.T @ X - kclass * (X.T @ MzX)
        bvec = X.T @ y - kclass * (X.T @ (y - Pzy))
    Ai = np.linalg.inv(A); b = Ai @ bvec; e = y - X @ b
    if kclass is None:
        Xh = PzX
    else:
        Xh = X - kclass * (X - PzX)
    if robust:
        V = (n / (n - k)) * Ai @ (Xh.T @ (Xh * (e ** 2)[:, None])) @ Ai
    else:
        V = (e @ e / (n - k)) * Ai
    return b, V, e

def liml(y, Y2, Z1, Z, robust=True):
    """LIML: Y2 endogenous (n x k2), Z1 included exogenous, Z full instrument matrix."""
    W = np.column_stack([y, Y2])
    def resid(A, B):
        return A - B @ np.linalg.lstsq(B, A, rcond=None)[0]
    M1W = resid(W, Z1); MZW = resid(W, Z)
    S1 = W.T @ M1W; SZ = W.T @ MZW
    kappa = np.min(np.real(np.linalg.eigvals(np.linalg.solve(SZ, S1))))
    X = np.column_stack([Y2, Z1])
    b, V, e = tsls(y, X, Z, robust=robust, kclass=kappa)
    return b, V, e, kappa

def first_stage_F(x, Z1, Z, robust=False):
    """F statistic for excluded instruments (columns of Z not in Z1) in regression of x on Z."""
    n = len(x)
    # detect excluded columns: those of Z after Z1 columns (caller builds Z = [Z1, Z2])
    k1 = Z1.shape[1]
    b, V, e = ols(x, Z, robust=robust)
    q = Z.shape[1] - k1
    d = b[k1:]; Vd = V[k1:, k1:]
    W = float(d @ np.linalg.solve(Vd, d))
    return W / q, q

def sargan(e, Z):
    n = len(e)
    Pe = Z @ np.linalg.lstsq(Z, e, rcond=None)[0]
    return float(e @ Pe / (e @ e / n))
