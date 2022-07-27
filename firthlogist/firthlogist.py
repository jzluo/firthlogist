import warnings
from copy import deepcopy
from math import sqrt

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
    pl_max_iter
        The maximum number of Newton-Raphson iterations for finding profile likelihood
        confidence intervals.
    pl_max_halfstep
        The maximum number of step-halvings in one iteration for finding profile
        likelihood confidence intervals.
    pl_max_stepsize
        The maximum step size while finding PL confidence intervals.
    tol
        Convergence tolerance for stopping.
    fit_intercept
        Specifies if intercept should be added.
    skip_lrt
        If True, p-values will not be calculated. Calculating the p-values can be
        time-consuming since the fitting procedure is repeated for each coefficient.
    skip_ci
        If True, confidence intervals will not be calculated. Calculating the confidence
        intervals via profile likelihoood is time-consuming.
    alpha
        Significance level (confidence interval = 1-alpha). 0.05 as default for 95% CI.

    Attributes
    ----------
    bse_
        Standard errors of the coefficients.
    classes_
        A list of the class labels.
    ci_
        The fitted profile likelihood confidence intervals.
    coef_
        The coefficients of the features.
    intercept_
        Fitted intercept. If `fit_intercept = False`, the intercept is set to zero.
    loglik_
        Fitted penalized log-likelihood.
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
        max_halfstep=25,
        max_stepsize=5,
        pl_max_iter=100,
        pl_max_halfstep=25,
        pl_max_stepsize=5,
        tol=0.0001,
        fit_intercept=True,
        skip_lrt=False,
        skip_ci=False,
        alpha=0.05,
    ):
        self.max_iter = max_iter
        self.max_stepsize = max_stepsize
        self.max_halfstep = max_halfstep
        self.pl_max_iter = pl_max_iter
        self.pl_max_halfstep = pl_max_halfstep
        self.pl_max_stepsize = pl_max_stepsize
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.skip_lrt = skip_lrt
        self.skip_ci = skip_ci
        self.alpha = alpha

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

        if not self.skip_ci:
            self.ci_ = np.column_stack(
                [
                    _profile_likelihood_ci(
                        X=X,
                        y=y,
                        side=side,
                        fitted_coef=self.coef_,
                        full_loglik=self.loglik_,
                        max_iter=self.pl_max_iter,
                        max_stepsize=self.pl_max_stepsize,
                        max_halfstep=self.pl_max_halfstep,
                        tol=self.tol,
                        alpha=0.05,
                    )
                    for side in [-1, 1]
                ]
            )

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
    warning_msg = (
        "Firth logistic regression failed to converge. Try increasing max_iter."
    )
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


def _get_aug_XW(X, preds, hats):
    rootW = np.sqrt(preds * (1 - preds) * (1 + hats))
    XW = rootW[:, np.newaxis] * X
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


def _predict(X, coef):
    preds = expit(X @ coef)
    np.clip(preds, a_min=1e-15, a_max=1 - 1e-15, out=preds)
    return preds


def _profile_likelihood_ci(
    X,
    y,
    side,
    fitted_coef,
    full_loglik,
    max_iter,
    max_stepsize,
    max_halfstep,
    tol,
    alpha,
):
    LL0 = full_loglik - chi2.ppf(1 - alpha, 1) / 2
    ci = []
    for coef_idx in range(fitted_coef.shape[0]):
        coef = deepcopy(fitted_coef)
        for iter in range(1, max_iter + 1):
            # preds = expit(X @ coef)
            preds = _predict(X, coef)
            loglike = -_loglikelihood(X, y, preds)
            XW = _get_XW(X, preds)
            hat = _hat_diag(XW)
            XW = _get_aug_XW(X, preds, hat)  # augmented data using hat diag
            fisher_info_mtx = XW.T @ XW
            U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
            # https://github.com/georgheinze/logistf/blob/master/src/logistf.c#L780-L781
            inv_fisher = np.linalg.pinv(fisher_info_mtx)
            tmp1x1 = U_star @ np.negative(inv_fisher) @ U_star
            underRoot = (
                -2 * ((LL0 - loglike) + 0.5 * tmp1x1) / (inv_fisher[coef_idx, coef_idx])
            )
            lambda_ = 0 if underRoot < 0 else side * sqrt(underRoot)
            U_star[coef_idx] += lambda_

            step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]
            mx = np.max(np.abs(step_size)) / max_stepsize
            if mx > 1:
                step_size = step_size / mx  # restrict to max_stepsize
            coef += step_size
            loglike_old = deepcopy(loglike)

            for halfs in range(1, max_halfstep + 1):
                # preds = expit(X @ coef)
                preds = _predict(X, coef)
                loglike = -_loglikelihood(X, y, preds)
                if (abs(loglike - LL0) < abs(loglike_old - LL0)) and loglike > LL0:
                    break
                step_size *= 0.5
                coef -= step_size
            if abs(loglike - LL0) <= tol:
                ci.append(coef[coef_idx])
                break
        if abs(loglike - LL0) > tol:
            ci.append(np.nan)
            warning_msg = (
                f"Non-converged PL confidence limits - max number of "
                f"iterations exceeded for variable x{coef_idx}. Try "
                f"increasing pl_max_iter."
            )
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
    return ci
