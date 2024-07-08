from sklearn.datasets import load_wine
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

wine = load_wine()

features, target = wine.data, wine.target

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    random_state=0
)

dummy = DummyRegressor(strategy="mean")

dummy.fit(features_train, target_train)

print(dummy.score(features_test, target_test))

from sklearn.linear_model import LinearRegression

ols = LinearRegression()

ols.fit(features_train, target_train)

print(ols.score(features_test, target_test))


from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

iris = load_iris()

features, target = iris.data, iris.target

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    random_state=0
)

dummy = DummyClassifier(strategy="uniform", random_state=1)

dummy.fit(features_train, target_train)

print(dummy.score(features_test, target_test))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(features_train, target_train)

print(classifier.score(features_test, target_test))

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=3, n_informative=3, n_redundant=0, n_classes=2, random_state=1)

logit = LogisticRegression()

print(cross_val_score(logit, X, y, scoring="accuracy"))

print(cross_val_score(logit, X, y, scoring="precision"))

print(cross_val_score(logit, X, y, scoring="recall"))

print(cross_val_score(logit, X, y, scoring="f1"))


from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


y_hat = logit.fit(X_train, y_train).predict(X_test)

print(accuracy_score(y_test, y_hat))