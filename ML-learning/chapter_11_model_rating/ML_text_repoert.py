from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()

features, target = iris.data, iris.target

class_names = iris.target_names

features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    random_state=0
)

classifier = LogisticRegression()

model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)

print(classification_report(target_test, target_predicted, target_names=class_names))
