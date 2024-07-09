from matplotlib import pyplot as plt
from pandas.core.common import random_state
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score

# Create feature matrix and target vector

features, target = make_classification(
    n_samples=10000,
    n_features=10,
    n_classes=2,
    n_informative=3,
    random_state=3
)

# Split into training and test sets

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.1,
    random_state=1
)

# Create classifier

logit = LogisticRegression()

# Train model

logit.fit(features_train, target_train)

# Get predicted probabilities

target_probabilities = logit.predict_proba(features_test)[:, 1]

# Create true and false positive rates

fp_rate, tp_rate, threashold = roc_curve(target_test, target_probabilities)

# Plot ROC curve

plt.title("Receiver Operating Characteristic")
plt.plot(fp_rate, tp_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# plt.show()

print(logit.predict_proba(features_test)[0:1])

features, target = make_classification(
    n_samples=10000,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_classes=3,
    random_state=1
)

logit = LogisticRegression()

print(cross_val_score(logit, features, target, scoring='accuracy'))
