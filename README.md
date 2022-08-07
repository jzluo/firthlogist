# firthlogist

[![PyPI](https://img.shields.io/pypi/v/firthlogist.svg)](https://pypi.org/project/firthlogist/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/firthlogist)
[![GitHub](https://img.shields.io/github/license/jzluo/firthlogist)](https://github.com/jzluo/firthlogist/blob/master/LICENSE)

A Python implementation of Logistic Regression with Firth's bias reduction.


## Installation
    pip install firthlogist

## Usage
firthlogist is sklearn compatible and follows the sklearn API.

```python
>>> from firthlogist import FirthLogisticRegression, load_sex2
>>> fl = FirthLogisticRegression()
>>> X, y, feature_names = load_sex2()
>>> fl.fit(X, y)
FirthLogisticRegression()
>>> fl.summary(xname=feature_names)
                 coef    std err     [0.025      0.975]      p-value
---------  ----------  ---------  ---------  ----------  -----------
age        -1.10598     0.42366   -1.97379   -0.307427   0.00611139
oc         -0.0688167   0.443793  -0.941436   0.789202   0.826365
vic         2.26887     0.548416   1.27304    3.43543    1.67219e-06
vicl       -2.11141     0.543082  -3.26086   -1.11774    1.23618e-05
vis        -0.788317    0.417368  -1.60809    0.0151846  0.0534899
dia         3.09601     1.67501    0.774568   8.03028    0.00484687
Intercept   0.120254    0.485542  -0.818559   1.07315    0.766584

Log-Likelihood: -132.5394
Newton-Raphson iterations: 8
```

### Parameters

`max_iter`: **_int_, default=25**

&emsp;The maximum number of Newton-Raphson iterations.

`max_halfstep`: **_int_, default=25**

&emsp;The maximum number of step-halvings in one Newton-Raphson iteration.

`max_stepsize`: **_int_, default=5**

&emsp;The maximum step size - for each coefficient, the step size is forced to
be less than max_stepsize.

`pl_max_iter`: **_int_, default=100**

&emsp;The maximum number of Newton-Raphson iterations for finding profile likelihood confidence intervals.

`pl_max_halfstep`: **_int_, default=25**

&emsp;The maximum number of step-halvings in one iteration for finding profile likelihood confidence intervals.

`pl_max_stepsize`: **_int_, default=5**

&emsp;The maximum step size while finding PL confidence intervals - for each coefficient, the step size is forced to
be less than max_stepsize.

`tol`: **_float_, default=0.0001**

&emsp;Convergence tolerance for stopping.

`fit_intercept`: **_bool_, default=True**

&emsp;Specifies if intercept should be added.

`skip_pvals`: **_bool_, default=False**

&emsp;If True, p-values will not be calculated. Calculating the p-values can
be expensive if `wald=False` since the fitting procedure is repeated for each
coefficient.

`skip_ci`: **_bool_, default=False**

&emsp;If True, confidence intervals will not be calculated. Calculating the confidence intervals via profile likelihoood is time-consuming.

`alpha`: **_float_, default=0.05**

&emsp;Significance level (confidence interval = 1-alpha). 0.05 as default for 95% CI.

`wald`: **_bool_, default=False**

&emsp;If True, uses Wald method to calculate p-values and confidence intervals.

`test_vars`: **Union[int, List[int]], default=None**

&emsp;Index or list of indices of the variables for which to calculate confidence intervals and p-values. If None, calculate for all variables. This option has no effect if `wald=True`.


### Attributes
`bse_`

&emsp;Standard errors of the coefficients.

`classes_`

&emsp;A list of the class labels.

`ci_`

&emsp; The fitted profile likelihood confidence intervals.

`coef_`

&emsp;The coefficients of the features.

`intercept_`

&emsp;Fitted intercept. If `fit_intercept = False`, the intercept is set to zero.

`loglik_`

&emsp;Fitted penalized log-likelihood.

`n_iter_`

&emsp;Number of Newton-Raphson iterations performed.

`pvals_`

&emsp;p-values calculated by penalized likelihood ratio tests.

## References
Firth, D (1993). Bias reduction of maximum likelihood estimates.
*Biometrika* 80, 27–38.

Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
regression. *Statistics in Medicine* 21: 2409-2419.
