from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()

features = digits.data

target = digits.target

standardizer = StandardScaler()

logit = LogisticRegression()

pipline = make_pipeline(standardizer, logit)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_results = cross_val_score(
    pipline,  # Pipeline
    features,  # Feature matrix
    target,  # Target vector
    cv=kf,  # Cross-validation technique
    scoring="accuracy",  # Loss function
    n_jobs=-1  # Use all CPU scores
)

print(cv_results.mean())  # oblicznie średniej z wyników

print(cv_results)  # wyniki dla każdego folda

from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.1,
    random_state=1
)

standardizer.fit(features_train)

features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

# pipline = make_pipeline(standardizer, logit)
#
# cv_results = cross_val_score(
#     pipline,  # Pipeline
#     features,  # Feature matrix
#     target,  # Target vector
#     cv=kf,  # Cross-validation technique
#     scoring="accuracy",  # Loss function
#     n_jobs=-1  # Use all CPU scores
# )
