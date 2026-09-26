"""Discrete choice models for the Koppelman data: conditional logit, nested logit, mixed logit,
multinomial probit (simple: iid N(0,1) errors via Gauss-Hermite; general: GHK)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import logsumexp

ALTS = ['train', 'air', 'bus', 'car']


def load(altvars=('cost', 'intime'), transform=None):
    d = pd.read_stata('Koppelman/Koppelman.dta')
    d['alternative'] = d['alternative'].astype(str)
    d = d.sort_values(['case', 'alternative'])
    cases = d.case.unique(); N = len(cases); J = 4
    d['j'] = d.alternative.map({a: i for i, a in enumerate(ALTS)})
    d = d.sort_values(['case', 'j'])
    X = np.stack([d[v].values.reshape(N, J) for v in altvars], axis=2).astype(float)  # N x J x K
    if transform:
        X = transform(X)
    y = d.choice.values.reshape(N, J).argmax(1)
    W = d.groupby('case')[['income', 'urban']].first().loc[cases].values.astype(float)
    W = np.column_stack([W, np.ones(N)])  # income, urban, const
    return X, W, y


def utilities(theta, X, W):
    """V_ij = X_ij'gamma + W_i'beta_j (beta_train = 0). theta = [gamma (K), beta_air(3), beta_bus(3), beta_car(3)]"""
    K = X.shape[2]
    g = theta[:K]; B = theta[K:K + 9].reshape(3, 3)  # rows air,bus,car ; cols income,urban,const
    V = X @ g
    V[:, 1:] += W @ B.T
    return V


def nll_clogit(theta, X, W, y):
    V = utilities(theta, X, W)
    return -(V[np.arange(len(y)), y] - logsumexp(V, axis=1)).sum()


def nll_nested(theta, X, W, y, groups, free_tau):
    """groups: list of lists of alt indices; free_tau: list of bools (tau estimated) per group."""
    K = X.shape[2]; V = utilities(theta[:K + 9], X, W)
    taus = []; it = K + 9
    for f in free_tau:
        if f: taus.append(theta[it]); it += 1
        else: taus.append(1.0)
    N = len(y); logP = np.zeros((N, 4)); IV = []
    for g, tau in zip(groups, taus):
        IV.append(logsumexp(V[:, g] / tau, axis=1))
    IV = np.array(IV).T  # N x G
    taus = np.array(taus)
    denom = logsumexp(IV * taus, axis=1)
    for gi, (g, tau) in enumerate(zip(groups, taus)):
        for k in g:
            logP[:, k] = V[:, k] / tau - IV[:, gi] + taus[gi] * IV[:, gi] - denom
    return -logP[np.arange(N), y].sum()


def halton(n, base):
    out = np.zeros(n)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base; r += f * (k % base); k //= base
        out[i] = r
    return out


def nll_mixed(theta, X, W, y, rand_idx, draws, lognormal=False):
    """random coefficient on X[:,:,rand_idx]: gamma = mu + sigma*z (normal) or exp(mu+sigma z) (lognormal)."""
    K = X.shape[2]; N = len(y); R = draws.shape[1]
    base = theta[:K + 9].copy(); mu = base[rand_idx]; sig = theta[K + 9]
    base[rand_idx] = 0.0
    V0 = utilities(base, X, W)  # N x J
    coef = np.exp(mu + sig * draws) if lognormal else mu + sig * draws  # N x R
    xr = X[:, :, rand_idx]  # N x J
    Vr = V0[:, :, None] + xr[:, :, None] * coef[:, None, :]  # N x J x R
    lp = Vr[np.arange(N), y, :] - logsumexp(Vr, axis=1)  # N x R
    return -(logsumexp(lp, axis=1) - np.log(R)).sum()


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)  # probabilists' Hermite: weight exp(-x^2/2)


def nll_mnp_simple(theta, X, W, y):
    """iid N(0,1) utility errors: P_j = int phi(u) prod_{l!=j} Phi(u + V_j - V_l) du."""
    V = utilities(theta, X, W); N = len(y)
    Vj = V[np.arange(N), y]
    diff = Vj[:, None] - V  # N x J, zero at chosen
    lp = np.zeros((N, len(GH_X)))
    for l in range(4):
        m = (np.arange(4) != l)
        # add log Phi for all alternatives except the chosen one
        term = norm.logcdf(GH_X[None, :] + diff[:, l][:, None])
        lp += np.where((y != l)[:, None], term, 0.0)
    P = (np.exp(lp) * GH_W[None, :]).sum(1) / np.sqrt(2 * np.pi)
    return -np.log(P).sum()


def num_hess(f, th, eps=1e-4):
    k = len(th); H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            hi = eps * max(1, abs(th[i])); hj = eps * max(1, abs(th[j]))
            e_i = np.zeros(k); e_j = np.zeros(k); e_i[i] = hi; e_j[j] = hj
            H[i, j] = (f(th + e_i + e_j) - f(th + e_i - e_j) - f(th - e_i + e_j) + f(th - e_i - e_j)) / (4 * hi * hj)
            H[j, i] = H[i, j]
    return H


def fit(f, th0, method='BFGS', maxiter=20000):
    r = minimize(f, th0, method=method, options={'maxiter': maxiter, 'gtol': 1e-7} if method == 'BFGS' else {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9})
    if method == 'BFGS':
        r2 = minimize(f, r.x, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
        r3 = minimize(f, r2.x, method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-8})
        r = r3 if r3.fun <= r.fun else r
    H = num_hess(f, r.x)
    V = np.linalg.inv(H)
    return r.x, np.sqrt(np.clip(np.diag(V), 0, None)), -r.fun


NAMES = ['cost', 'intime', 'air_inc', 'air_urb', 'air_c', 'bus_inc', 'bus_urb', 'bus_c', 'car_inc', 'car_urb', 'car_c']


def report(lab, th, se, ll, names=None):
    names = names or NAMES
    print(lab, 'LL %.1f' % ll, ' '.join('%s=%.4f(%.4f)' % (n, t, s) for n, t, s in zip(names, th, se)), flush=True)


def halton_fast(n, base, skip=10):
    idx = np.arange(skip + 1, skip + n + 1); out = np.zeros(n); f = 1.0 / base; i = idx.copy()
    while np.any(i > 0):
        out += f * (i % base); i //= base; f /= base
    return out


def sigma_struct(p):
    """Stata-style structural covariance: train var 1, uncorrelated; air var 1 (scale alternative);
    p = [log sd_bus, log sd_car, c1, c2, c3] with the correlation matrix of (air,bus,car) from a
    spherical-type Cholesky parameterization."""
    sd = np.array([1.0, np.exp(p[0]), np.exp(p[1])])
    L = np.zeros((3, 3)); L[0, 0] = 1
    t1, t2, t3 = np.tanh(p[2]), np.tanh(p[3]), np.tanh(p[4])
    L[1, 0] = t1; L[1, 1] = np.sqrt(1 - t1 ** 2)
    L[2, 0] = t2; L[2, 1] = t3 * np.sqrt(1 - t2 ** 2); L[2, 2] = np.sqrt(max(1 - t2 ** 2 - (t3 ** 2) * (1 - t2 ** 2), 1e-12))
    Rm = L @ L.T
    S = np.zeros((4, 4)); S[0, 0] = 1.0; S[1:, 1:] = np.outer(sd, sd) * Rm
    return S


def nll_mnp_general(theta, X, W, y, U):
    """GHK simulated likelihood. U: N x R x 3 uniform (Halton) draws."""
    K = X.shape[2]; V = utilities(theta[:K + 9], X, W); S = sigma_struct(theta[K + 9:])
    N, R = U.shape[0], U.shape[1]; P = np.zeros(N)
    for j in range(4):
        ix = np.where(y == j)[0]
        if len(ix) == 0: continue
        others = [l for l in range(4) if l != j]
        A = np.zeros((3, 4)); A[:, j] = -1
        for r_, l in enumerate(others): A[r_, l] = 1
        Om = A @ S @ A.T; L = np.linalg.cholesky(Om)
        b = V[ix, j][:, None] - V[ix][:, others]  # eta_l < b_l
        prob = np.ones((len(ix), R)); z = np.zeros((len(ix), R, 3))
        for k in range(3):
            mu = (z[:, :, :k] * L[k, :k]).sum(2) if k > 0 else 0.0
            up = norm.cdf((b[:, k][:, None] - mu) / L[k, k])
            prob *= up
            if k < 2:
                z[:, :, k] = norm.ppf(np.clip(U[ix, :, k] * up, 1e-300, 1 - 1e-16))
        P[ix] = prob.mean(1)
    return -np.log(np.maximum(P, 1e-300)).sum()


def fit_quick(f, th0, hess=True):
    r = minimize(f, th0, method='BFGS', options={'maxiter': 5000, 'gtol': 1e-6})
    if not hess:
        return r.x, None, -r.fun
    H = num_hess(f, r.x); V = np.linalg.pinv(H)
    return r.x, np.sqrt(np.clip(np.diag(V), 0, None)), -r.fun


def sigma_chol(p):
    """structural covariance: train var 1, uncorrelated; (air,bus,car) block = L L' with L11 = 1 (air scale)."""
    L = np.array([[1.0, 0, 0], [p[0], p[1], 0], [p[2], p[3], p[4]]])
    S = np.zeros((4, 4)); S[0, 0] = 1.0; S[1:, 1:] = L @ L.T
    return S


def nll_mnp_general2(theta, X, W, y, U):
    K = X.shape[2]; S = sigma_chol(theta[K + 9:])
    V = utilities(theta[:K + 9], X, W); N, R = U.shape[0], U.shape[1]; P = np.zeros(N)
    for j in range(4):
        ix = np.where(y == j)[0]
        if len(ix) == 0: continue
        others = [l for l in range(4) if l != j]
        A = np.zeros((3, 4)); A[:, j] = -1
        for r_, l in enumerate(others): A[r_, l] = 1
        Om = A @ S @ A.T + 1e-10 * np.eye(3); L = np.linalg.cholesky(Om)
        b = V[ix, j][:, None] - V[ix][:, others]
        prob = np.ones((len(ix), R)); z = np.zeros((len(ix), R, 3))
        for k in range(3):
            mu = (z[:, :, :k] * L[k, :k]).sum(2) if k > 0 else 0.0
            up = norm.cdf((b[:, k][:, None] - mu) / L[k, k])
            prob *= up
            if k < 2:
                z[:, :, k] = norm.ppf(np.clip(U[ix, :, k] * up, 1e-300, 1 - 1e-16))
        P[ix] = prob.mean(1)
    return -np.log(np.maximum(P, 1e-300)).sum()


def omega_diff(p):
    """unrestricted covariance of w=(e_air-e_train, e_bus-e_train, e_car-e_train), Omega=LL', L11=sqrt(2)."""
    L = np.array([[np.sqrt(2.0), 0, 0], [p[0], p[1], 0], [p[2], p[3], p[4]]])
    return L @ L.T


def nll_mnp_general3(theta, X, W, y, U):
    K = X.shape[2]; Om = omega_diff(theta[K + 9:])
    V = utilities(theta[:K + 9], X, W); N, R = U.shape[0], U.shape[1]; P = np.zeros(N)
    for j in range(4):
        ix = np.where(y == j)[0]
        if len(ix) == 0: continue
        others = [l for l in range(4) if l != j]
        # eta_l = (e_l - e_j) expressed in w (w index a=l-1 for l>=1; w for train = 0)
        B = np.zeros((3, 3))
        for r_, l in enumerate(others):
            if l >= 1: B[r_, l - 1] += 1
            if j >= 1: B[r_, j - 1] -= 1
        C = B @ Om @ B.T + 1e-10 * np.eye(3); L = np.linalg.cholesky(C)
        b = V[ix, j][:, None] - V[ix][:, others]
        prob = np.ones((len(ix), R)); z = np.zeros((len(ix), R, 3))
        for k in range(3):
            mu = (z[:, :, :k] * L[k, :k]).sum(2) if k > 0 else 0.0
            up = norm.cdf((b[:, k][:, None] - mu) / L[k, k])
            prob *= up
            if k < 2:
                z[:, :, k] = norm.ppf(np.clip(U[ix, :, k] * up, 1e-300, 1 - 1e-16))
        P[ix] = prob.mean(1)
    return -np.log(np.maximum(P, 1e-300)).sum()
