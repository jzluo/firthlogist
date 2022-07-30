from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

from firthlogist import FirthLogisticRegression, load_endometrial, load_sex2

TEST_DIR = Path(__file__).parent


@parametrize_with_checks([FirthLogisticRegression()])
def test_estimator(estimator, check):
    return check(estimator)


@pytest.mark.parametrize(
    "loader_func, data_shape, target_shape, n_target, xname",
    [
        (load_sex2, (239, 6), (239,), (2,), ["age", "oc", "vic", "vicl", "vis", "dia"]),
        (load_endometrial, (79, 3), (79,), (2,), ["NV", "PI", "EH"]),
    ],
)
def test_loaders(loader_func, data_shape, target_shape, n_target, xname):
    X, y, feature_names = loader_func()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == data_shape
    assert y.shape == target_shape
    assert np.unique(y).shape == n_target
    assert feature_names == xname


@pytest.fixture
def sex2():
    # compare with logistf for logistf::sex2
    X = np.loadtxt(TEST_DIR / "sex2.csv", delimiter=",", skiprows=1)
    y = X[:, 0]
    X = X[:, 1:]
    data = {
        "X": X,
        "y": y,
        "logistf_coef": np.array(
            [-1.10598131, -0.06881673, 2.26887464, -2.11140817, -0.78831694, 3.09601166]
        ),
        "logistf_intercept": 0.12025405,
        "logistf_n_iter": 8,
    }
    return data


@pytest.fixture
def data(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("data", ["sex2"], indirect=True)
def test_compare_to_logistf(data):
    firth = FirthLogisticRegression(fit_intercept=True)
    firth.fit(data["X"], data["y"])
    assert_allclose(firth.coef_, data["logistf_coef"], rtol=1e-05)
    assert_allclose(firth.intercept_, data["logistf_intercept"], rtol=1e-05)
    assert firth.n_iter_ == data["logistf_n_iter"]
