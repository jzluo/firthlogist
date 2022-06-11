import warnings

import numpy as np
from scipy.special import expit
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

    Attributes
    ----------
    classes_
        A list of the class labels.
    coef_
        The coefficients of the features.
    intercept_
        Fitted intercept. If `fit_intercept = False`, the intercept is set to zero.
    n_iter_
        Number of Newton-Raphson iterations performed.

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
    ):
        self.max_iter = max_iter
        self.max_stepsize = max_stepsize
        self.max_halfstep = max_halfstep
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _more_tags(self):
        return {"binary_only": True}

    def _validate_input(self, X, y):
        if self.max_iter < 0:
            raise ValueError(
                f"Maximum number of iterations must be positive; "
                f"got max_iter={self.max_iter}"
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

        self.coef_, self.n_iter_ = _firth_newton_raphson(
            X, y, self.max_iter, self.max_stepsize, self.max_halfstep, self.tol
        )

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


def _firth_newton_raphson(X, y, max_iter, max_stepsize, max_halfstep, tol):
    # see logistf reference manual for explanation of procedure
    coef = np.zeros(X.shape[1])
    for iter in range(1, max_iter + 1):
        preds = expit(X @ coef)
        XW = _get_XW(X, preds)
        fisher_info_mtx = XW.T @ XW
        hat = _hat_diag(XW)
        U_star = np.matmul(X.T, y - preds + np.multiply(hat, 0.5 - preds))
        step_size = np.linalg.lstsq(fisher_info_mtx, U_star, rcond=None)[0]

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
                return coef_new, iter

        if iter > 1 and np.linalg.norm(coef_new - coef) < tol:
            return coef_new, iter

        coef += step_size

    warning_msg = "Firth logistic regression failed to converge."
    warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
    return coef, max_iter


def _loglikelihood(X, y, preds):
    # penalized log-likelihood
    XW = _get_XW(X, preds)
    fisher_info_mtx = XW.T @ XW
    penalty = 0.5 * np.log(np.linalg.det(fisher_info_mtx))
    return -1 * (np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) + penalty)


def _get_XW(X, preds):
    rootW = np.sqrt(preds * (1 - preds))
    XW = rootW[:, np.newaxis] * X
    return XW


def _hat_diag(XW):
    # Get diagonal elements of the hat matrix
    Q = np.linalg.qr(XW, mode="reduced")[0]
    hat = np.einsum("ij,ij->i", Q, Q)
    return hat
