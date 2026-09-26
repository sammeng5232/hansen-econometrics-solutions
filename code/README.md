# Code for the empirical exercises (Chapters 8–29)

Python scripts behind the numbers, tables and figures in the empirical
solutions of Chapters 8–29. Chapters 2–7 carry their R code inside the solutions.
The estimators are written out directly: OLS/IV/GMM, VARs, unit-root and
cointegration tests, dynamic-panel GMM, kernel and series regression, RDD, NLLS,
quantile regression, probit/logit, multinomial choice models, Tobit/CLAD,
information criteria and the Lasso. The only dependencies are

```
numpy  scipy  pandas  matplotlib
```

(tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, matplotlib 3.11). No statsmodels or scikit-learn.

## Data layout

The data are not redistributed. Download them from the textbook site
(<https://www.ssc.wisc.edu/~bhansen/econometrics/>) into a working directory
laid out like this:

```
data/
  cps09mar/cps09mar.txt
  AB1991/AB1991.dta        AJR2001/AJR2001.dta    AK1991/AK1991.dta
  BMN2016/BMN2016.dta      CK1994/CK1994.dta      Card1995/Card1995.dta
  DDK2011/DDK2011.dta      DS2004/DS2004.dta      Invest1993/Invest1993.dta
  Koppelman/Koppelman.dta  MRW1992/MRW1992.dta    Nerlove1963/Nerlove1963.dta
  RR2010/RR2010.dta
  progs/                   # the textbook's "Econometrics Programs" bundle; used for
                           # FRED-MD.dta, FRED-QD.dta, Kilian2009.dta, CHJ2004.dta,
                           # AL1999.dta, LM2007.dta, PSS2017.dta
```

Data paths are relative to the current directory, so run each script **from
`data/`**:

```bash
cd data
python ../code/ch17emp.py
```

Local helper modules load from `code/`. Figures are written to `../figures/`
relative to the script, i.e. the repository's `figures/` folder.

## Helper modules

| Module | Contents |
|---|---|
| `lib.py`, `boot.py` | OLS with robust covariance, Wald tests; jackknife and bootstrap (percentile, BC, BCa) |
| `ivlib.py` | 2SLS, LIML, first-stage F, Sargan test |
| `varlib.py` | VAR estimation, AIC, impulse responses (recursive and structural), bootstrap bands |
| `urlib.py` | ADF, KPSS, Johansen trace test with interpolated critical values |
| `dpd2.py` | Arellano–Bond / Blundell–Bond GMM (sparse), Windmeijer-corrected SEs |
| `felib.py` | Within transformation, fixed effects and difference-in-differences with clustered SEs |
| `kreg.py` | Nadaraya–Watson and local linear regression, ROT and CV bandwidths |
| `ch20lib.py` | Polynomial and spline series regression, CV / AIC, HC3 |
| `choice.py` | Conditional, nested and mixed logit; simple (Gauss–Hermite) and general (GHK) multinomial probit |

## Scripts by exercise

| Chapter | Exercises | Script(s) |
|---|---|---|
| 8 | 8.19 | `ch8emp.py` |
| 9 | 9.24–9.29 | `mrw_mc.py` (9.24 Monte Carlo), `ch9emp.py` (+ `t42.py`) |
| 10 | 10.28–10.31 | `ch10emp.py`, `ddkboot.py` |
| 12 | 12.22–12.27 | `ajr.py`, `card.py`, `ak.py` |
| 13 | 13.27–13.28 | `gmm13.py` |
| 14 | 14.18–14.22 | `ts14.py` |
| 15 | 15.14–15.20 | `ch15emp.py` |
| 16 | 16.12–16.14 | `rep16.py` (replicates the chapter's unit-root examples), `ch16emp.py` |
| 17 | 17.15–17.18 | `val17.py` (replicates the chapter's dynamic-panel estimates), `ch17emp.py` … `ch17emp4.py` |
| 18 | 18.6–18.8 | `rep18.py`, `rep18b.py` (replicate the chapter), `ch18emp.py` |
| 19 | 19.7–19.11 | `ch19emp.py` |
| 20 | 20.9–20.18 | `ch20emp.py`, `ch20emp_b.py` |
| 21 | 21.5–21.9 | `ch21emp.py` |
| 23 | 23.8–23.10 | `ch23emp.py` |
| 24 | 24.13–24.16 | `ch24emp.py` |
| 25 | 25.15–25.19 | `ch25emp.py` |
| 26 | 26.12–26.14 | `ch26mnl.py` |
| 26 | 26.15–26.16, simple probit | `ch26kop.py` |
| 26 | 26.17 | `ml_run.py base`, `ml_run.py time`, `ml_run.py lognormal` |
| 26 | 26.18 | `mnp_run3.py base`, `mnp_run3.py log` |
| 27 | 27.9–27.11 | `ch27emp.py` |
| 28 | 28.12 | `ch28emp.py` |
| 29 | 29.9–29.10 | `ch29emp.py` |

## Run times

Most scripts finish in seconds to a few minutes. The slow ones are:

- the Chapter 17 GMM scripts;
- the cross-validation searches in `ch19emp.py`;
- `ml_run.py` (mixed logit, simulated ML with 200–400 Halton draws);
- `mnp_run3.py` (general multinomial probit, GHK with 300 Halton draws; up to an hour per specification);
- `ch29emp.py` (10-fold CV Lasso paths by coordinate descent; about half an hour).

Simulation-based estimates (mixed logit, general probit) use deterministic
Halton sequences, so reruns reproduce the reported numbers. They differ from
Stata's `cmmixlogit` / `cmmprobit` output only by simulation error.

## Not included

Three figures were drawn interactively and have no script here: the FRED-MD series plots
`ch16_12` and `ch16_14`, and the loss-function illustration `ch22_g`.
