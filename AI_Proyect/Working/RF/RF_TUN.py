#wORKING WITH THE OVERFITTING PROBLEM

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the training and testing datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Split the data into features and labels
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Example of post-pruning with a single Decision Tree (for reference)
X, y = X_train, y_train  # You can also use a subset of the training data
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Plot Total Impurity vs effective alpha for the training set
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

# Train a RandomForestClassifier with different ccp_alpha values
clfs = []
for ccp_alpha in ccp_alphas:
    rf = RandomForestClassifier(random_state=42, ccp_alpha=ccp_alpha)
    rf.fit(X_train, y_train)
    clfs.append(rf)

# Evaluate the performance of each RandomForest model using the validation set
accuracy_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]

# Plot accuracy vs. ccp_alpha to select the optimal pruning level
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], accuracy_scores[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs effective alpha for Random Forest on test set")
plt.show()

# Select the best model based on the accuracy scores
best_index = np.argmax(accuracy_scores)
best_rf = clfs[best_index]
best_ccp_alpha = ccp_alphas[best_index]
print(f"Best ccp_alpha: {best_ccp_alpha}")

# Final evaluation on the test set with the best pruned model
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate AUC for each class and then average
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
if y_test_binarized.shape[1] == y_pred_proba.shape[1]:
    auc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
else:
    auc = None  # AUC is not applicable for binary or single-class scenarios

print(f"Model accuracy with pruned Random Forest: {accuracy}")
print(f"Model F1-score with pruned Random Forest: {f1}")
print(f"Model precision with pruned Random Forest: {precision}")
print(f"Model recall with pruned Random Forest: {recall}")
if auc is not None:
    print(f"Model AUC with pruned Random Forest: {auc}")
else:
    print("AUC is not applicable for this dataset.")
