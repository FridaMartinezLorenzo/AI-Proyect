import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_Standard.csv')
test_data = pd.read_csv('../TrainTest/Split/test_Standard.csv')
# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']

X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# Impute missing values in the features with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Determine the nu parameter as the proportion of outliers
nu = y_train.value_counts()[1] / y_train.shape[0]

# Train the One-Class SVM model
oc_svm = OneClassSVM(kernel='rbf', nu=nu, gamma=10000)
oc_svm.fit(X_train)

# Prediction on the training set using One-Class SVM
y_pred_train = oc_svm.predict(X_train)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_train_bin = np.where(y_pred_train == -1, 0, 1)

# Evaluate the One-Class SVM model on the training set
print("Training Set Metrics:")
print("Accuracy:", accuracy_score(y_train, y_pred_train_bin))
print("Precision:", precision_score(y_train, y_pred_train_bin, zero_division=0))
print("Recall:", recall_score(y_train, y_pred_train_bin, zero_division=0))
print("F1-Score:", f1_score(y_train, y_pred_train_bin, zero_division=0))
print("Area Under Curve (AUC):", roc_auc_score(y_train, y_pred_train_bin))
print("\nClassification Report for Training Set:")
print(classification_report(y_train, y_pred_train_bin, zero_division=0))


# Prediction on the test set using One-Class SVM
y_pred_test = oc_svm.predict(X_test)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

# Evaluate the One-Class SVM model on the test set
print("\nTest Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_test_bin))
print("Precision:", precision_score(y_test, y_pred_test_bin, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_test_bin, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred_test_bin, zero_division=0))
print("Area Under Curve (AUC):", roc_auc_score(y_test, y_pred_test_bin))
print("\nClassification Report for Test Set:")
print(classification_report(y_test, y_pred_test_bin, zero_division=0))