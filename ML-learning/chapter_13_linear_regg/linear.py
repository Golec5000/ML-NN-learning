from scipy.odr import polynomial
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1
)

regresion = LinearRegression()

model = regresion.fit(features, target)

print("Intercept: ", model.intercept_)

print("Coef: ", model.coef_)

print(target[0])

print(model.predict(features)[0])

print(model.score(features, target))

from sklearn.preprocessing import PolynomialFeatures

features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1
)

interaction = PolynomialFeatures(
    degree=3,
    include_bias=False,
    interaction_only=True
)

features_interaction = interaction.fit_transform(features)

model = regresion.fit(features_interaction, target)

print(features[0])

import numpy as np

interaction_term = np.multiply(features[:, 0], features[:, 1])

print(interaction_term[0])

features, target = make_regression(
    n_samples=100,
    n_features=3,
    n_informative=2,
    n_targets=1,
    noise=0.2,
    coef=False,
    random_state=1
)

polynomial = PolynomialFeatures(degree=3, include_bias=False)
polynomial_features = polynomial.fit_transform(features)

regresion = LinearRegression()

model = regresion.fit(polynomial_features, target)

print(polynomial_features[0])
