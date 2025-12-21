import pytest
import numpy as np
from regression_inference import *

STD_TOL = 1e-10
WEAK_TOL = 1e-5

'''
Perfect Fit Expectations:


    The model should explain 100% of the variance in the 
    training set.

        - R-squared, R-squared-adj == 1


    The model should have no residuals.

        - MSE = 0


    The model should have explained the total sum of squared errors.

        - ESS = TSS


    The model should have no residuals.

        - RSS = 0


    The specified coefficients should be recovered exactly.

        - theta = [5, 2, -1, 3]


General OLS Expectations:


    Residuals should be Orthogonal 

        - model.X.T @ model.residuals = 0


    Predicting the features at the sample mean returns y_bar

        - model.predict([mean, mean, mean]) = y_bar


    Predicting the features at 0 returns the intercept

        - model.predict([0,0,0]) = Intercept


    Confidence intervals for coefficient estimates should be accurate

        - model.ci_high > model.ci_low


    Model fitted with perfect multicollinearity either raises ValueError,
    or fits with np.linalg.det(XTX) ~= 0

        - X3 = 2 * X1 + 3 * X2
    
'''

@pytest.fixture
def perfect_fit():

    n_samples, n_features = 10, 3

    X_features = np.random.randint(0, 10, size=(n_samples, n_features))
    X = np.hstack([np.ones((n_samples, 1)), X_features])

    y = X @ np.array([5, 2, -1, 3])

    return LinearRegression().fit(X, y, alpha=0.05)


def test_perfect_r_squared(perfect_fit):

    r_sq = perfect_fit.r_squared

    assert r_sq == 1.0, (

        (f"Expected r-squared = 1, Got: r-squared = {r_sq}")
    )


def test_perfect_r_squared_adj(perfect_fit):

    r_sq = perfect_fit.r_squared_adjusted

    assert r_sq == 1.0, (

        (f"Expected r-squared-adj = 1, Got: r-squared_adj = {r_sq}")
    )


def test_perfect_TSS_ESS(perfect_fit):

    tss = perfect_fit.tss
    ess = perfect_fit.ess

    assert np.allclose(tss, ess), (

        (f"Expected TSS = ESS, Got: {tss} != {ess}")
    )


def test_perfect_RSS(perfect_fit):

    rss = perfect_fit.rss

    assert np.allclose(rss, 0), (

        (f"Expected RSS = 0, Got: {rss} != 0")
    )


def test_perfect_MSE(perfect_fit):

    mse = perfect_fit.mse

    assert np.allclose(mse, 0), (

        (f"Expected MSE = 0, Got: {mse} != 0")
    )


def test_perfect_ci_coefficients(perfect_fit):

    ci_low = perfect_fit.ci_low
    ci_high = perfect_fit.ci_high

    assert np.allclose(ci_low, ci_high), (

        (f"Expected ci_high = ci_low, Got: {ci_high} != {ci_low}")
    )


def test_perfect_feature_recovery(perfect_fit):

    known_coefficients = [5, 2, -1, 3]
    model_coefficients = perfect_fit.theta

    assert np.allclose(model_coefficients, known_coefficients), (

        (f"Expected theta = {known_coefficients}, Got: {model_coefficients}")
    )




''' General Linear Model Expectations '''


@pytest.fixture
def random_fit():

    n_samples, n_features = 10, 3

    X_features = np.random.randint(0, 10, size = (n_samples, n_features))

    X = np.hstack([np.ones((n_samples, 1)), X_features])

    y = np.random.randint(0, 10, size = (n_samples))

    return LinearRegression().fit(X, y, alpha=0.05)


def test_random_orthogonality(random_fit):

    x_transpose = random_fit.X.T
    residuals = random_fit.residuals

    assert np.allclose(x_transpose @ residuals, 0, atol = STD_TOL), (

        (f"Expected XT @ Residuals = 0, Got: {x_transpose @ residuals}")
    )


def test_random_mean_prediction(random_fit):

    X = random_fit.X.T
    y = random_fit.y

    coef1 = X[1]
    coef2 = X[2]
    coef3 = X[3]

    prediction = random_fit.predict([np.mean(coef1),np.mean(coef2),np.mean(coef3)])

    assert np.allclose(prediction, np.mean(y)), (

        (f"Expected prediction = {np.mean(y)}, Got: {prediction}")
    )


def test_random_zero_prediction(random_fit):

    intercept = random_fit.intercept
    prediction = random_fit.predict([0,0,0])

    assert np.allclose(intercept, prediction), (

        (f"Expected prediction = {intercept}, Got: {prediction}")
    )


def test_random_ci_coefficients(random_fit):

    ci_low = random_fit.ci_low
    ci_high = random_fit.ci_high

    assert np.all(ci_high > ci_low), (

        (f"Expected ci_high > ci_low, Got: {ci_high} !> {ci_low}")
    )



'''
Perfect Multicollinearity

Passes: ValueError raised during fit, does not fit the model.

Passes with warning: ValueError not raised from sufficient tol in pseudo random number generation,
fitted model, determinant of XTX approx. 0

Fails: Determinant of XTX != 0
'''

def test_perfect_multicollinearity():

    n_samples, n_features = 10, 3

    X_features = (
        np.random.randint(0, 10, size = (n_samples, n_features - 1))
    )
    
    X_perfect_multicollinearity = (
        np.hstack([X_features, 2*X_features[:, 0:1] + 3*X_features[:, 1:2]])
    )
    
    X = (
        np.hstack([np.ones((n_samples, 1)), X_perfect_multicollinearity])
    )

    y = (
        np.random.randint(0, 10, size = (n_samples))
    )
    
    try:
        LinearRegression().fit(X, y)

    except ValueError:

        XTX = X.T @ X
        det = np.linalg.det(XTX)

        assert np.allclose(det, 0, atol = WEAK_TOL), (

            (f"Expected Determinant of X.T @ X = 0, Got: {det}")
        )
        






