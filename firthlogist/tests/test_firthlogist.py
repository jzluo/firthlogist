from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

from firthlogist import FirthLogisticRegression

TEST_DIR = Path(__file__).parent


@parametrize_with_checks([FirthLogisticRegression()])
def test_estimator(estimator, check):
    return check(estimator)


@pytest.fixture
def diabetes():
    # compare with logistf for diabetes data
    X = np.loadtxt(TEST_DIR / "diabetes.csv", delimiter=",", skiprows=1)
    y = X[:, -1]
    X = X[:, :-1]
    data = {
        "X": X,
        "y": y,
        "logistf_coef": np.array(
            [
                0.1215056439,
                0.0345600170,
                -0.0130517652,
                0.0005825059,
                -0.0011697657,
                0.0879587577,
                0.9286920256,
                0.0147477743,
            ]
        ),
        "logistf_intercept": -8.2661614602,
        "logistf_n_iter": 6,
    }
    return data


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


@pytest.mark.parametrize("data", ["diabetes", "sex2"], indirect=True)
def test_compare_to_logistf(data):
    firth = FirthLogisticRegression(fit_intercept=True)
    firth.fit(data["X"], data["y"])
    assert_allclose(firth.coef_, data["logistf_coef"], rtol=1e-05)
    assert_allclose(firth.intercept_, data["logistf_intercept"], rtol=1e-05)
    assert firth.n_iter_ == data["logistf_n_iter"]
