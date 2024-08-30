import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_Standard.csv')
test_data = pd.read_csv('../TrainTest/Split/test_Standard.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Initialize the Local Outlier Factor model with novelty detection enabled
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)

# Fit the model on the training data
lof.fit(X_train)

# Predict on the test data
y_pred_test = lof.predict(X_test)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

# Evaluate the model on the test data
print("Accuracy on the test set using Local Outlier Factor:")
print(accuracy_score(y_test, y_pred_test_bin))

print("\nClassification report for Local Outlier Factor on test set:")
print(classification_report(y_test, y_pred_test_bin))
