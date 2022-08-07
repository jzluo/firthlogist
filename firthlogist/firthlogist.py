import warnings
from copy import deepcopy
from importlib.resources import open_text
from math import sqrt

import numpy as np
from scipy.linalg import lapack
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tabulate import tabulate


class FirthLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression with Firth's bias reduction method.

    This is based on the implementation in the `logistf` R package. Please see the
    `logistf` reference and Heinze & Schemper (2002) for details about the procedure.

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
    skip_pvals
        If True, p-values will not be calculated. Calculating the p-values can be
        time-consuming if `wald=False` since the fitting procedure is repeated for each
        coefficient.
    skip_ci
        If True, confidence intervals will not be calculated. Calculating the confidence
        intervals via profile likelihoood is time-consuming.
    alpha
        Significance level (confidence interval = 1-alpha). 0.05 as default for 95% CI.
    wald
        If True, uses Wald method to calculate p-values and confidence intervals.
    test_vars
        Index or list of indices of the variables for which to calculate confidence
        intervals and p-values. If None, calculate for all variables. This option has
        no effect if wald=True.

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
        max_halfstep=0,
        max_stepsize=5,
        pl_max_iter=100,
        pl_max_halfstep=0,
        pl_max_stepsize=5,
        tol=0.0001,
        fit_intercept=True,
        skip_pvals=False,
        skip_ci=False,
        alpha=0.05,
        wald=False,
        test_vars=None,
    ):
        self.max_iter = max_iter
        self.max_stepsize = max_stepsize
        self.max_halfstep = max_halfstep
        self.pl_max_iter = pl_max_iter
        self.pl_max_halfstep = pl_max_halfstep
        self.pl_max_stepsize = pl_max_stepsize
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.skip_pvals = skip_pvals
        self.skip_ci = skip_ci
        self.alpha = alpha
        self.wald = wald
        self.test_vars = test_vars

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
            if not self.wald:
                self.ci_ = _profile_likelihood_ci(
                    X=X,
                    y=y,
                    fitted_coef=self.coef_,
                    full_loglik=self.loglik_,
                    max_iter=self.pl_max_iter,
                    max_stepsize=self.pl_max_stepsize,
                    max_halfstep=self.pl_max_halfstep,
                    tol=self.tol,
                    alpha=self.alpha,
                    test_vars=self.test_vars,
                )
            else:
                self.ci_ = _wald_ci(self.coef_, self.bse_, self.alpha)
        else:
            self.ci_ = np.full((self.coef_.shape[0], 2), np.nan)

        # penalized likelihood ratio tests
        if not self.skip_pvals:
            if not self.wald:
                self.pvals_ = _penalized_lrt(
                    self.loglik_,
                    X,
                    y,
                    self.max_iter,
                    self.max_stepsize,
                    self.max_halfstep,
                    self.tol,
                    self.test_vars,
                )

            else:
                self.pvals_ = _wald_test(self.coef_, self.bse_)
        else:
            self.pvals_ = np.full(self.coef_.shape[0], np.nan)

        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0

        return self

    def summary(self, xname=None, tablefmt="simple"):
        """
        Prints a summary table.

        Parameters
        ----------
        xname
            Names for the X variables. Default is x1, x2, ... Must match the number of
            parameters in the model.
        tablefmt
            `tabulate` table format for output. Please see the documentation for
            `tabulate` for options.
        """
        check_is_fitted(self)
        if xname and len(xname) != len(self.coef_):
            raise ValueError(
                f"Length of xname ({len(xname)}) does not match the number of "
                f"parameters in the model ({len(self.coef_)})"
            )

        if not xname:
            xname = [f"x{i}" for i in range(1, len(self.coef_) + 1)]

        coef = list(self.coef_)
        if self.fit_intercept:
            xname.append("Intercept")
            coef.append(self.intercept_)

        headers = [
            "",
            "coef",
            "std err",
            f"[{self.alpha/2}",
            f"{1-self.alpha/2}]",
            "p-value",
        ]
        table = zip(xname, coef, self.bse_, self.ci_[:, 0], self.ci_[:, 1], self.pvals_)
        table = tabulate(table, headers, tablefmt=tablefmt)
        table += "\n\n"
        table += f"Log-Likelihood: {round(self.loglik_, 4)}\n"
        table += f"Newton-Raphson iterations: {self.n_iter_}\n"
        print(table)
        if self.fit_intercept:
            xname.pop()
        return

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
    if mask is not None:
        XW[:, mask] = 0
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


def _penalized_lrt(
    full_loglik, X, y, max_iter, max_stepsize, max_halfstep, tol, test_vars
):
    if test_vars is None:
        test_var_indices = range(X.shape[1])
    elif isinstance(test_vars, int):  # single index
        test_var_indices = [test_vars]
    else:  # list, tuple, or set of indices
        test_var_indices = sorted(test_vars)

    pvals = []
    for mask in test_var_indices:
        _, null_loglik, _ = _firth_newton_raphson(
            X,
            y,
            max_iter,
            max_stepsize,
            max_halfstep,
            tol,
            mask,
        )
        pvals.append(_lrt(full_loglik, null_loglik))
    if len(pvals) < X.shape[1]:
        pval_array = np.full(X.shape[1], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            pval_array[test_var_idx] = pvals[idx]
        return pval_array
    return np.array(pvals)


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
    fitted_coef,
    full_loglik,
    max_iter,
    max_stepsize,
    max_halfstep,
    tol,
    alpha,
    test_vars,
):
    LL0 = full_loglik - chi2.ppf(1 - alpha, 1) / 2
    lower_bound = []
    upper_bound = []
    if test_vars is None:
        test_var_indices = range(fitted_coef.shape[0])
    elif isinstance(test_vars, int):  # single index
        test_var_indices = [test_vars]
    else:  # list, tuple, or set of indices
        test_var_indices = sorted(test_vars)
    for side in [-1, 1]:
        # for coef_idx in range(fitted_coef.shape[0]):
        for coef_idx in test_var_indices:
            coef = deepcopy(fitted_coef)
            for iter in range(1, max_iter + 1):
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
                    -2
                    * ((LL0 - loglike) + 0.5 * tmp1x1)
                    / (inv_fisher[coef_idx, coef_idx])
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
                    preds = _predict(X, coef)
                    loglike = -_loglikelihood(X, y, preds)
                    if (abs(loglike - LL0) < abs(loglike_old - LL0)) and loglike > LL0:
                        break
                    step_size *= 0.5
                    coef -= step_size
                if abs(loglike - LL0) <= tol:
                    if side == -1:
                        lower_bound.append(coef[coef_idx])
                    else:
                        upper_bound.append(coef[coef_idx])
                    break
            if abs(loglike - LL0) > tol:
                if side == -1:
                    lower_bound.append(np.nan)
                else:
                    upper_bound.append(np.nan)
                warning_msg = (
                    f"Non-converged PL confidence limits - max number of "
                    f"iterations exceeded for variable x{coef_idx}. Try "
                    f"increasing pl_max_iter."
                )
                warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
    bounds = np.column_stack([lower_bound, upper_bound])
    if len(lower_bound) < fitted_coef.shape[0]:
        ci = np.full([fitted_coef.shape[0], 2], np.nan)
        for idx, test_var_idx in enumerate(test_var_indices):
            ci[test_var_idx] = bounds[idx]
        return ci

    return bounds


def _wald_ci(coef, bse, alpha):
    lower_ci = coef + norm.ppf(alpha / 2) * bse
    upper_ci = coef + norm.ppf(1 - alpha / 2) * bse
    return np.column_stack([lower_ci, upper_ci])


def _wald_test(coef, bse):
    # 1 - pchisq((beta^2/vars), 1), in our case bse = vars^0.5
    return chi2.sf(np.square(coef) / np.square(bse), 1)


def load_sex2():
    """
    Load the sex2 dataset from `logistf`.

    Returns
    -------
    X
        sex2 data as numpy array
    y
        sex2 `case` target column
    feature_names
        List of feature names

    References
    ----------
    Cytel Inc., (2010) LogXact 9 user manual, Cambridge, MA:Cytel Inc
    """
    with open_text("firthlogist.datasets", "sex2.csv") as sex2:
        X = np.loadtxt(sex2, skiprows=1, delimiter=",")
    y = X[:, 0]
    X = X[:, 1:]
    feature_names = ["age", "oc", "vic", "vicl", "vis", "dia"]
    return X, y, feature_names


def load_endometrial():
    """
    Load the endometrial cancer dataset analyzed in Heinze and Schemper (2002). The data
    was originally provided by Dr E. Asseryanis from the Vienna University Medical
    School

    Returns
    -------
    X
        endometrial data as numpy array
    y
        endometrial `HG` target column
    feature_names
        List of feature names

    References
    ----------
    Agresti, A (2015). Foundations of Linear and Generalized Linear Models.
    Wiley Series in Probability and Statistics.

    Heinze G, Schemper M (2002). A solution to the problem of separation in logistic
    regression. Statistics in Medicine 21: 2409-2419.
    """
    with open_text("firthlogist.datasets", "endometrial.csv") as sex2:
        X = np.loadtxt(sex2, skiprows=1, delimiter=",")
    y = X[:, -1]
    X = X[:, :-1]
    feature_names = ["NV", "PI", "EH"]
    return X, y, feature_names
