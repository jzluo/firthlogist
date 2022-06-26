# firthlogist

[![PyPI](https://img.shields.io/pypi/v/firthlogist.svg)](https://pypi.org/project/firthlogist/)
[![GitHub](https://img.shields.io/github/license/jzluo/firthlogist)](https://github.com/jzluo/firthlogist/blob/master/LICENSE)

A Python implementation of Logistic Regression with Firth's bias reduction.

WIP!

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
bse = firth.bse_
```

## References
Firth, D (1993). Bias reduction of maximum likelihood estimates.
*Biometrika* 80, 27–38.

Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
regression. *Statistics in Medicine* 21: 2409-2419.
