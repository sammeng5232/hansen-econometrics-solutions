import numpy as np, pandas as pd
"""glmnet-style Lasso: minimize (1/2n)||y - a - Xb||^2 + lam*||b||_1 with standardized X (population sd),
coordinate descent with warm starts on a decreasing lambda path; 10-fold CV; lambda.min and lambda.1se."""


def lasso_path(X, y, lams, tol=1e-10, maxit=200000):
    """covariance-update coordinate descent on standardized regressors (Gram matrix, O(p) per update)."""
    n, p = X.shape; mu = X.mean(0); sd = X.std(0); keep = np.where(sd > 0)[0]
    Z = (X[:, keep] - mu[keep]) / sd[keep]
    G = Z.T @ Z / n; cvec = Z.T @ (y - y.mean()) / n
    q = len(keep); b = np.zeros(q); Gb = np.zeros(q); out = []
    for lam in lams:
        for it in range(maxit):
            maxd = 0.0
            for j in range(q):
                rho = cvec[j] - Gb[j] + b[j]          # G[j,j] = 1
                nb = np.sign(rho) * max(abs(rho) - lam, 0.0)
                d = nb - b[j]
                if d != 0.0:
                    Gb += G[:, j] * d; b[j] = nb; maxd = max(maxd, abs(d))
            if maxd < tol: break
        coef = np.zeros(p); coef[keep] = b / sd[keep]
        a = y.mean() - mu @ coef
        out.append((a, coef.copy()))
    return out


def cv_lasso(X, y, nlam=100, K=10, seed=1):
    n, p = X.shape; sd = X.std(0); keep = sd > 0
    Z = np.zeros_like(X); Z[:, keep] = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    lmax = np.max(np.abs(Z.T @ (y - y.mean())) / n); lams = lmax * np.logspace(0, -4, nlam)
    folds = np.random.default_rng(seed).permutation(n) % K
    err = np.zeros((K, nlam))
    for k in range(K):
        tr = folds != k; te = ~tr
        path = lasso_path(X[tr], y[tr], lams)
        for l, (a, b) in enumerate(path):
            err[k, l] = np.mean((y[te] - a - X[te] @ b) ** 2)
    cvm = err.mean(0); cvse = err.std(0, ddof=1) / np.sqrt(K)
    imin = int(np.argmin(cvm)); i1se = int(np.where(cvm <= cvm[imin] + cvse[imin])[0].min())
    full = lasso_path(X, y, lams)
    return lams, cvm, cvse, imin, i1se, full


c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c.columns = ['age', 'female', 'hisp', 'education', 'earnings', 'hours', 'week', 'union', 'uncov', 'region', 'race', 'marital']
c['lw'] = np.log(c.earnings / (c.hours * c.week)); c['exp'] = (c.age - c.education - 6) / 40


def design(s):
    cols = {'education': s.education.astype(float)}
    for v in (12, 13, 14, 15, 16, 18, 20):
        cols['edu%d' % v] = (s.education == v).astype(float)
    for j in range(1, 10):
        cols['exp^%d' % j] = s.exp ** j
    cols['married'] = s.marital.isin([1, 2, 3]).astype(float); cols['divorced'] = (s.marital == 5).astype(float)
    cols['separated'] = (s.marital == 6).astype(float); cols['widowed'] = (s.marital == 4).astype(float); cols['never'] = (s.marital == 7).astype(float)
    for r, nm in zip((1, 2, 3, 4), ('Northeast', 'Midwest', 'South', 'West')):
        cols[nm] = (s.region == r).astype(float)
    cols['union'] = s.union.astype(float)
    D = pd.DataFrame(cols); return D.columns.tolist(), D.values


for lab, s in [('Asian women', c[(c.female == 1) & (c.race == 4)]), ('Hispanic men', c[(c.female == 0) & (c.hisp == 1)])]:
    names, X = design(s); y = s.lw.values
    lams, cvm, cvse, imin, i1se, full = cv_lasso(X, y)
    print('=====', lab, 'n', len(y), 'zero-variance columns:', [nm for nm, v in zip(names, X.std(0)) if v == 0])
    for tag, i in [('lambda.min', imin), ('lambda.1se', i1se)]:
        a, b = full[i]; nz = np.abs(b) > 1e-10
        print('  %s = %.5f  CV MSE %.4f (se %.4f)  nonzero %d' % (tag, lams[i], cvm[i], cvse[i], nz.sum()))
        print('     intercept %.4f; ' % a + '; '.join('%s %.4f' % (nm, bb) for nm, bb, z in zip(names, b, nz) if z))
    # OLS for comparison (drop collinear: edu15, never, West)
    keepc = [j for j, nm in enumerate(names) if nm not in ('edu15', 'never', 'West')]
    Xo = np.column_stack([np.ones(len(y)), X[:, keepc]])
    bo = np.linalg.lstsq(Xo, y, rcond=None)[0]; eo = y - Xo @ bo
    print('  OLS (all regressors) residual var %.4f, k=%d' % (eo @ eo / len(y), Xo.shape[1]))
    # implied experience profile m(30)-m(0) under lambda.min
    a, b = full[imin]; ex_idx = [names.index('exp^%d' % j) for j in range(1, 10)]
    prof = lambda e: sum(b[ex_idx[j - 1]] * (e / 40) ** j for j in range(1, 10))
    print('  lambda.min: log-wage difference exp 30 vs 0: %.3f; 10 vs 0: %.3f' % (prof(30), prof(10)))
    a, b = full[i1se]; print('  lambda.1se: exp 30 vs 0: %.3f' % sum(b[ex_idx[j - 1]] * (30 / 40) ** j for j in range(1, 10)))
