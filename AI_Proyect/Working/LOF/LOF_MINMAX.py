import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label'].values
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label'].values

# Initialize the Local Outlier Factor model with novelty detection enabled
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.15, novelty=True)

# Fit the model on the training data using values
lof.fit(X_train.values, y_train)

# Predict on the test data using values
y_pred_test = lof.predict(X_test.values)

print("Predictions on the test set using Local Outlier Factor:")
print(y_pred_test)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

print("\nBinary predictions on the test set using Local Outlier Factor:")
print(y_pred_test_bin)

print("\nActual labels on the test set:")
print(y_test)

# Evaluate the model on the test data
print("Accuracy on the test set using Local Outlier Factor:")
print(accuracy_score(y_test, y_pred_test_bin))

print("\nClassification report for Local Outlier Factor on test set:")
print(classification_report(y_test, y_pred_test_bin))
