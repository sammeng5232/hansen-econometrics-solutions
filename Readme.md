# Solutions to Bruce E. Hansen, *Econometrics* (Princeton University Press, 2022)

Authors: Zhizhong Pu (Harvard University), Zijun Meng, and Claude Opus 5

## Authorship and contributions

- **Zhizhong Pu** wrote the original solutions to Chapters 2, 3, 4, 5 and 7
  (the upstream project, March 2025), distributed as separate `ChapN.tex` files.
- **Zijun Meng and Claude Opus 5** produced everything else in this edition:
  consolidated the chapters into a single document, completed the previously
  unanswered exercises, corrected errors in the earlier drafts (see "Notes on
  this revision" below), added the new Chapters 8–29, and wrote all the Python
  code in `code/` and the figures in `figures/`.

## Contents

Every exercise in the book has a worked solution: 486 exercises in 27 chapters.
Chapters 1 and 6 have no exercises. **The exercise statements are not
reproduced here: read them in the book.** Each solution is headed by the number
of the exercise it answers and uses the notation of that exercise. Numbered
results cited as "Theorem 2.4.4" or "(4.32)" refer to Hansen (2022).

| File | Contents |
|---|---|
| `Hansen_Econometrics_Solutions.tex` / `.pdf` | all the solutions in one document (232 pages; solutions only, without the exercise statements): title page, linked table of contents, PDF bookmarks for every chapter and exercise |
| `figures/` | figures included by Chapters 15–26 |
| `code/` | Python code for the empirical exercises of Chapters 8–29 (see `code/README.md`) |

| Chapter | Exercises |
|---|---|
| 2. Conditional Expectation and Projection | 2.1 – 2.22 |
| 3. The Algebra of Least Squares | 3.1 – 3.26 |
| 4. Least Squares Regression | 4.1 – 4.26 |
| 5. Normal Regression | 5.1 – 5.12 |
| 7. Asymptotic Theory for Least Squares | 7.1 – 7.28 |
| 8. Restricted Estimation | 8.1 – 8.22 |
| 9. Hypothesis Testing | 9.1 – 9.29 |
| 10. Resampling Methods | 10.1 – 10.31 |
| 11. Multivariate Regression | 11.1 – 11.15 |
| 12. Instrumental Variables | 12.1 – 12.27 |
| 13. Generalized Method of Moments | 13.1 – 13.28 |
| 14. Time Series | 14.1 – 14.22 |
| 15. Multivariate Time Series | 15.1 – 15.20 |
| 16. Non-Stationary Time Series | 16.1 – 16.14 |
| 17. Panel Data | 17.1 – 17.18 |
| 18. Difference in Differences | 18.1 – 18.8 |
| 19. Nonparametric Regression | 19.1 – 19.11 |
| 20. Series Regression | 20.1 – 20.18 |
| 21. Regression Discontinuity | 21.1 – 21.9 |
| 22. M-Estimators | 22.1 – 22.4 |
| 23. Nonlinear Least Squares | 23.1 – 23.10 |
| 24. Quantile Regression | 24.1 – 24.16 |
| 25. Binary Choice | 25.1 – 25.19 |
| 26. Multiple Choice | 26.1 – 26.18 |
| 27. Censoring and Selection | 27.1 – 27.11 |
| 28. Model Selection, Stein Shrinkage, and Model Averaging | 28.1 – 28.12 |
| 29. Machine Learning | 29.1 – 29.10 |

## Building

The `.tex` file is self-contained: the shared notation and the exercise
environment are defined in its preamble. It only needs the `figures/` folder
next to it. Compile from this directory with

```bash
pdflatex Hansen_Econometrics_Solutions.tex
```

Run it two or three times so that the contents page numbers and PDF bookmarks
settle.

Microtype's font expansion is switched off in the preamble. With it on, pdfTeX
occasionally placed part of a line off the page.

## Empirical exercises

The empirical exercises use the data sets from the textbook website
(<https://www.ssc.wisc.edu/~bhansen/econometrics/>): `cps09mar`, `DDK2011`,
`Nerlove1963`, `MRW1992`, `Invest1993`, `AJR2001`, `Card1995`, `AK1991`,
`FRED-QD`, `FRED-MD`, `Kilian2009`, `AB1991`, `CK1994`, `DS2004`, `BMN2016`,
`RR2010`, `CHJ2004`, `AL1999`, `LM2007`, `PSS2017` and `Koppelman`. The data are
not redistributed here.

- **Chapters 3–7:** the solutions include the R code that produced the numbers.
- **Chapters 8–29:** the Python scripts are in `code/`, with a map from exercise to script.

Where the chapter reports estimates on the same data, the solutions first
reproduce the book's own tables and equations as a check on sample definitions
and methods. Any discrepancy is stated in the solution. Examples include
(3.49) and (4.60), Table 16.2, (17.114), the Chapter 18 tables and regressions,
Table 21.1, Table 23.1, Table 24.2, Table 26.1 and the Chapter 28
model-selection table. Table 17.3 is matched to within 0.002.

Estimates of the mixed logit and the general multinomial probit are simulated
maximum likelihood with Halton draws. They agree with the book to within
simulation error.

## Notes on this revision

This edition completes the previously unanswered exercises and corrects a number
of errors in the earlier drafts. The substantive corrections to Chapters 2–7 are:

- **2.2** — sign error in the CEF error, `e = Y − a + bX` → `e = Y − a − bX`.
- **2.16** — the best linear predictor was computed with the no-intercept
  formula `E[XY]/E[X²]`. With an intercept in the model the slope is
  `cov(X,Y)/var(X) = −15/73` and the intercept is `55/73`; the reported `15/16`
  was wrong in both formula and arithmetic.
- **3.3, 3.4, 3.6, 3.11** — transposes, identity dimensions and a garbled final
  line repaired.
- **3.14** — the recursive least squares formula is now proved (Sherman–Morrison),
  rather than left incomplete.
- **4.1(c,d)** — the variance of the `k`-th moment estimator is
  `(μ_2k − μ_k²)/n`, finite iff `E[Y^2k] < ∞`. The earlier answer invoked a
  condition `2k < n`, which is not what is required, and the proposed variance
  estimator omitted the `1/n`.
- **4.9** — `(1 − h_ii)^{-1}` was taken outside the sum; it depends on `i`.
- **4.11** — the stated result is `E[V^HC2 | X] = σ²(X'X)^{-1}`, not `σ²`.
- **4.15** — i.i.d. sampling does not imply homoskedasticity, so
  `var(e|X) = σ²I` cannot be assumed; parts (b) and (c) turn on exactly that
  point and are now answered.
- **5.1** — `Q = Σ Z_i²`, not `Σ Z_i`.
- **5.5** — the variance of the fitted values is `P var(e|X) P' = σ²P`.
- **7.1** — matrix-transpose conventions repaired; the condition for consistency
  is `E[X₁X₂']β₂ = 0`, which is weaker than `E[X₁X₂'] = 0` or `β₂ = 0`.
- **7.6** — the method of moments estimator of `Ω` must use residuals, not the
  unobservable errors; the formula for `β̂` was missing an inverse.
- **7.8** — the earlier derivation applied the CLT to a non-i.i.d. average and
  left the variance unevaluated. The correct limit is
  `√n(σ̂² − σ²) →d N(0, E[e⁴] − σ⁴)`, obtained by showing the two
  estimation-error terms are `o_p(1)`.

Chapters 8–29 are new in this edition.

## Apparent errors in the textbook

These are flagged where they arise. They are listed here for convenience.

**Exercise statements and formulas**

- **(5.20)** — the `(2,2)` entry of the information matrix should be `n/(2σ⁴)`.
  Its inverse, `2σ⁴/n`, is printed in both displays.
- **4.20** — the second display asks for `E[(β̂ − β̃)(β̂ − β)']`, where
  `(β̂ − β̃)(β̂ − β̃)'` appears to be intended.
- **10.13** — "b" where `c` is intended.
- **11.12** — the hint writes `Σ^{1/2}` twice; the second is `Σ^{−1/2}`.
- **11.13** — the limit is the SUR variance `V*_β`, not `V_β`.
- **12.10** — the control-function regression is `Y = X'β + u'γ + ν`, not `Y = Z'β + u'γ + ν`.
- **14.1** — `ρ̂(k) →p ρ(k)`, not `γ(k)`.
- **14.7** — the upper limit of the denominator is `∞`, not `q`.
- **14.10** — an AR(2) has `α₂Y_{t−2}`, not `α₂Y_{t−1}`.
- **14.11** — the second model should read `Y_t = μ + u_t`.
- **15.12** — the `(2,2)` entry should be `σ₂²`, not `σ₁²`.
- **16.12** — the FRED-MD mnemonic for initial claims is `claimsx`.
- **17.1** — in (17.11)–(17.12) the sums run over `i = 1,…,N`, and (17.12) omits `σ²_ε`.
- **17.3** — the claim is false for an individual `t` (a counterexample is given); it holds on average.
- **17.5** — the text's reference to "Exercise 17.28" means this exercise.
- **17.8** — the sum in (17.37) runs over `N`.
- **17.11** — in (17.58) the coefficient on `B̂_fe` should be `1/(T−2)`, not `1/(T−1)`.
- **18.5** — the periods 2001–2010 and 2010–2020 overlap in 2010.
- **20.4** — the quadratic spline term is `β₂x²`, not `β₂x³`.
- **20.18** — the denominator in (20.42) is `1 + ⌊(enrollment − 1)/40⌋`, and the class-size effect involves `β₇`, not `β₄`.
- **23.10** — "much above `β₇`" should read "much above `γ`".
- **24.11** — `Ω̂_τ` in Section 24.8 should be scaled by `1/n`, not `1/h`.
- **24.12** — `h(0, X₂, U)` should read `h(0, X, U)`.
- **25.14** — a binary choice model has `Y = 1{Y* > 0}`. The printed `Y = Y*·1{Y* > 0}` is a Tobit model.
- **27.3** — "consistent for `β̂`" should read "for `β`".
- **29.1** — in the leave-one-out formula, `Σ_{j≠i} X_j Y_i` should be `Σ_{j≠i} X_j Y_j`.

**Reported empirical results**

- **(18.8)** — we obtain `+0.005 (0.005)` for *OffOutFlows*, not `−0.005`.
- **(21.5)** — the reported estimates and `n = 482` correspond to the window `±8`,
  not `±13.8`. The book's code computes `h₀ = h√3` but selects the sample with `h`.
- **23.8** — the standard error of `σ = 1/(1 − ρ)` in the CES example is `0.71`.
  The printed `0.46` omits a factor `(1 − ρ̂)^{−1}`.
- **Figure 25.1** — the book's code counts widowed respondents as married.
- **Table 26.1**
  - Nested logit: the cost coefficient is `−0.001`, not `−0.011`, and air × urban is `−0.28`, not `+0.28`.
  - Mixed logit: the mean intime coefficient is `−0.017`, not `−0.014`.
- **Chapter 28 table**
  - FIC\* for models 1–3 are 126, 77, 84 rather than 86, 48, 53.
  - AIC and BIC omit the constant `n` and count `K = k`.

**Data notes**

- In `cps09mar` no worker has exactly 15 years of education, so the `edu = 15`
  dummy in Exercises 29.9–29.10 is identically zero.
