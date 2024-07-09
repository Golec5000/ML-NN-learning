from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load the iris dataset

iris = datasets.load_iris()
features, target = iris.data, iris.target

decisiontree = DecisionTreeClassifier(random_state=0)

model = decisiontree.fit(features, target)

observation = [[5, 4, 3, 2]]

print(model.predict(observation))

print(model.predict_proba(observation))

decisiontree_entropy = DecisionTreeClassifier(
    criterion='entropy', random_state=0
)

model_entropy = decisiontree_entropy.fit(features, target)

print(model_entropy.predict(observation))

print(model_entropy.predict_proba(observation))

diabets = datasets.load_diabetes()
features, target = diabets.data, diabets.target

decisiontree = DecisionTreeClassifier(random_state=0)

model = decisiontree.fit(features, target)

observation = [features[0]]

print(model.predict(observation))

import pydotplus
from IPython.display import Image
from sklearn import tree

iris = datasets.load_iris()

features, target = iris.data, iris.target

decisiontree = DecisionTreeClassifier(random_state=0)

model = decisiontree.fit(features, target)

dot_data = tree.export_graphviz(
    decisiontree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
