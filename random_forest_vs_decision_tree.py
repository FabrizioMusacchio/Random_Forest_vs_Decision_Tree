"""
A simple script to compare the performance of a decision tree classifier and a random forest classifier.

Author: Fabrizio Musacchio (fabriziomusacchio.com)
Date: June 20, 2023

For reproducibility:

conda create -n random_forest_vs_decision_tree -y python=3.9
conda activate random_forest_vs_decision_tree
conda install -y mamba
mamba install -y scikit-learn matplotlib numpy ipykernel pandas scikit-image napari[all]
"""
# %% IMPORTS
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# %% MAIN
# load the Iris dataset:
data = load_iris()
X, y = data.data, data.target

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# decision tree classifier:
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Visualizing the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_classifier, filled=True)
plt.show()


# random forest classifier:
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# get feature importances
importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# convert feature names to a list:
feature_names = list(data.feature_names)

# plot feature importances:
plt.figure(figsize=(3, 4))
plt.bar(range(X.shape[1]), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance - Random Forest')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.xaxis.set_ticks_position('none')
plt.show()

# calculate accuracy for decision tree
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)

# calculate accuracy for random forest
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
# %% END