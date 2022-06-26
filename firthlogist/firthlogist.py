import warnings

import numpy as np
from scipy.linalg import lapack
from scipy.special import expit
from scipy.stats import chi2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Firth's bias reduction method.

    This is based on the implementation in the logistf R package. Please see the
    logistf reference and Heinze & Schemper (2002) for details about the procedure.

    Parameters
    ----------
    max_iter
        The maximum number of Newton-Raphson iterations.
    max_halfstep
        The maximum number of step-halvings in one Newton-Raphson iteration.
    max_stepsize
        The maximum step size - for each coefficient, the step size is forced to
        be less than max_stepsize.
    tol
        Convergence tolerance for stopping.
    fit_intercept
        Specifies if intercept should be added.
    skip_lrt
        If True, p-values will not be calculated. Calculating the p-values can be
        expensive since the fitting procedure is repeated for each coefficient.

    Attributes
    ----------
    bse_
        Standard errors of the coefficients.
    classes_
        A list of the class labels.
    coef_
        The coefficients of the features.
    intercept_
        Fitted intercept. If `fit_intercept = False`, the intercept is set to zero.
    n_iter_
        Number of Newton-Raphson iterations performed.
    pvals_
        p-values calculated by penalized likelihood ratio tests.

    References
    ----------
    Firth D (1993). Bias reduction of maximum likelihood estimates.
    Biometrika 80, 27–38.

    Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
    regression. Statistics in Medicine 21: 2409-2419.
    """

    def __init__(
        self,
        max_iter=25,
        max_halfstep=1000,
        max_stepsize=5,
        tol=0.0001,
        fit_intercept=True,
        skip_lrt=False,
    ):
        self.max_iter = max_iter
        self.max_stepsize = max_stepsize
        self.max_halfstep = max_halfstep
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.skip_lrt = skip_lrt

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        if self.max_iter < 0:
            raise ValueError(
                f"Maximum number of iterations must be positive; "
                f"got max_iter={self.max_iter}"
            )
        if self.max_halfstep < 0:
            raise ValueError(
                f"Maximum number of step-halvings must >= 0; "
                f"got max_halfstep={self.max_iter}"
            )
        if self.tol < 0:
            raise ValueError(
                f"Tolerance for stopping criteria must be positive; got tol={self.tol}"
            )
        X, y = self._validate_data(X, y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Got {len(self.classes_)} - only 2 classes supported.")
        y = LabelEncoder().fit_transform(y).astype(X.dtype, copy=False)

        return X, y

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        self.coef_, self.loglik_, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol
        )

        self.bse_ = _bse(X, self.coef_)

        # penalized likelihood ratio tests
        if not self.skip_lrt:
            pvals = []
            # mask is 1-indexed because of `if mask` check in _get_XW()
            for mask in range(1, self.coef_.shape[0] + 1):
                _, null_loglik, _ = _firth_newton_raphson(
                    X,
                    y,
                    self.max_iter,
                    self.max_stepsize,
                    self.max_halfstep,
                    self.tol,
                    mask,
                )
                pvals.append(_lrt(self.loglik_, null_loglik))
            self.pvals_ = np.array(pvals)

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        scores = X @ self.coef_ + self.intercept_
        return scores

    def predict(self, X):
        decision = self.decision_function(X)
        if len(decision.shape) == 1:
            indices = (decision > 0).astype(int)
        else:
            indices = decision.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        decision = self.decision_function(X)
        if decision.ndim == 1:
            decision = np.c_[-decision, decision]
        proba = expit(decision)
        return proba


def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol, mask=None):
    # see logistf reference manual for explanation of procedure
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        preds = expit(X @ coef)
        XW = _get_XW(X, preds, mask)

        fisher_info_mtx = XW.T @ XW
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
        # if mask:
        #     step_size[mask] = 0

        # step-halving
        mx = np.max(np.abs(step_size)) / max_stepsize
        if mx > 1:
            step_size = step_size / mx  # restrict to max_stepsize
        coef_new = coef + step_size
        preds_new = expit(X @ coef_new)
        loglike = _loglikelihood(X, y, preds)
        loglike_new = _loglikelihood(X, y, preds_new)
        steps = 0
        while loglike < loglike_new:
            step_size *= 0.5
            coef_new = coef + step_size
            preds_new = expit(X @ coef_new)
            loglike_new = _loglikelihood(X, y, preds_new)
            steps += 1
            if steps == max_halfstep:
                warning_msg = "Step-halving failed to converge."
                warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
                return coef_new, -loglike_new, iter

        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, -loglike_new, iter

        coef += step_size
    warning_msg = "Firth logistic regression failed to converge."
    warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
    return coef, -loglike_new, max_iter


def _loglikelihood(X, y, preds):
    # penalized log-likelihood
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    return -1 * (np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) + penalty)


def _get_XW(X, preds, mask=None):
    # mask is 1-indexed because 0 == None
    rootW = np.sqrt(preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X

    # is this equivalent??
    # https://github.com/georgheinze/logistf/blob/master/src/logistf.c#L150-L159
    if mask:
        XW[:, mask - 1] = 0
    return XW


def _hat_diag(XW):
    # Get diagonal elements of the hat matrix
    # Q = np.linalg.qr(XW, mode="reduced")[0]
    qr, tau, _, _ = lapack.dgeqrf(XW)
    Q, _, _ = lapack.dorgqr(qr, tau)
    hat = np.einsum("ij,ij->i", Q, Q)
    return hat


def _bse(X, coefs):
    # se in logistf is diag(object$var) ^ 0.5, where var is the covariance matrix,
    # which is the inverse of the observed fisher information matrix
    # https://stats.stackexchange.com/q/68080/343314
    preds = expit(X @ coefs)
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW
    return np.sqrt(np.diag(np.linalg.pinv(fisher_info_mtx)))


def _lrt(full_loglik, null_loglik):
    # in logistf: 1-pchisq(2*(fit.full$loglik-fit.i$loglik),1)
    lr_stat = 2 * (full_loglik - null_loglik)
    p_value = chi2.sf(lr_stat, df=1)
    return p_value
