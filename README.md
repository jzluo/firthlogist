# firthlogist

[![PyPI](https://img.shields.io/pypi/v/firthlogist.svg)](https://pypi.org/project/firthlogist/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/firthlogist)
[![GitHub](https://img.shields.io/github/license/jzluo/firthlogist)](https://github.com/jzluo/firthlogist/blob/master/LICENSE)

A Python implementation of Logistic Regression with Firth's bias reduction.


## Installation
    pip install firthlogist

## Usage
firthlogist follows the sklearn API.

```python
from firthlogist import FirthLogisticRegression

firth = FirthLogisticRegression()
firth.fit(X, y)
coefs = firth.coef_
pvals = firth.pvals_
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

`skip_lrt`: **_bool_, default=False**

&emsp;If True, p-values will not be calculated. Calculating the p-values can
be expensive since the fitting procedure is repeated for each
coefficient.

`skip_ci`: **_bool_, default=False**

&emsp;If True, confidence intervals will not be calculated. Calculating the confidence intervals via profile likelihoood is time-consuming.


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
