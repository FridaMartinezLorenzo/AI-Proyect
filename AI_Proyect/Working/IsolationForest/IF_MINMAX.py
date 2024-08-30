import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# Fit the model on the training data
iso_forest.fit(X_train)

# Predict on the training data
y_pred_train = iso_forest.predict(X_train)

# Predict on the test data
y_pred_test = iso_forest.predict(X_test)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_train_bin = np.where(y_pred_train == -1, 0, 1)
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

# Evaluate the model on the training data
print("Accuracy on the training set using Isolation Forest:")
print(accuracy_score(y_train, y_pred_train_bin))

print("\nClassification report for Isolation Forest on training set:")
print(classification_report(y_train, y_pred_train_bin))

# Evaluate the model on the test data
print("Accuracy on the test set using Isolation Forest:")
print(accuracy_score(y_test, y_pred_test_bin))

print("\nClassification report for Isolation Forest on test set:")
print(classification_report(y_test, y_pred_test_bin))
