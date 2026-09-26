import numpy as np, pandas as pd
c = pd.read_csv('cps09mar/cps09mar.txt', sep=r'\s+', header=None)
c.columns = ['age', 'female', 'hisp', 'education', 'earnings', 'hours', 'week', 'union', 'uncov', 'region', 'race', 'marital']
c['lw'] = np.log(c.earnings / (c.hours * c.week)); c['exp'] = c.age - c.education - 6


def models(s, married_def=3):
    n = len(s); y = s.lw.values
    base = [np.ones(n), (s.marital <= married_def).astype(float)] + [(s.region == r).astype(float) for r in (2, 3, 4)]
    e = s.education.values.astype(float); x = s.exp.values.astype(float) / 10
    edu = {'College': [(e >= 16).astype(float)], 'Spline': [e, np.maximum(e - 9, 0)],
           'Dummy': [(e == v).astype(float) for v in (12, 13, 14, 16, 18, 20)]}
    out = []
    for p in (2, 4, 6):
        for lab in ('College', 'Spline', 'Dummy'):
            X = np.column_stack(base + edu[lab] + [x ** j for j in range(1, p + 1)])
            G = np.zeros(X.shape[1]); G[-p:] = [3.0 ** j for j in range(1, p + 1)]  # m(30)-m(0) with exp/10
            out.append((lab, p, X, G))
    res = []
    for lab, p, X, G in out:
        n, k = X.shape
        XXi = np.linalg.inv(X.T @ X); b = XXi @ X.T @ y; e_ = y - X @ b
        h = np.einsum('ij,jk,ik->i', X, XXi, X)
        s2 = e_ @ e_ / n; K = k + 1
        bic = n + n * np.log(2 * np.pi * s2) + K * np.log(n); aic = n + n * np.log(2 * np.pi * s2) + 2 * K
        cv = np.sum((e_ / (1 - h)) ** 2)
        V = XXi @ ((X * e_[:, None]).T @ (X * e_[:, None])) @ XXi * n / (n - k)  # HC1
        mu = G @ b; se = np.sqrt(G @ V @ G)
        res.append(dict(lab=lab, p=p, mu=mu, se=se, bic=bic, aic=aic, cv=cv, n=n))
    mu_full = res[-1]['mu']
    for r in res:
        r['fic'] = r['n'] * (r['mu'] - mu_full) ** 2 + 2 * r['n'] * r['se'] ** 2
    return res


def table(res, title):
    print(title, 'n =', res[0]['n'])
    for key, fmt in [('mu', '%5.0f%%'), ('se', '%5.0f '), ('bic', '%6.0f'), ('aic', '%6.0f'), ('cv', '%6.0f'), ('fic', '%6.0f')]:
        vals = [r[key] * (100 if key in ('mu', 'se') else 1) for r in res]
        best = '' if key in ('mu', 'se') else '  min: model %d' % (int(np.argmin(vals)) + 1)
        print('%-4s' % key, ' '.join(fmt % v for v in vals), best)


s = c[(c.female == 1) & (c.race == 4)]
table(models(s), 'Asian women (Table 28.1 replication)')
s = c[(c.female == 1) & (c.hisp == 1)]
res = models(s)
table(res, 'Hispanic women')
