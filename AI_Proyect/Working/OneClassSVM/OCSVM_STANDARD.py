import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../dataset_pre_processed_standard.csv')

# Extract features and labels from train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']

# Impute missing values in the features with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Train the One-Class SVM model
oc_svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
oc_svm.fit(X_train)

# Prediction on the training set using One-Class SVM
y_pred_train = oc_svm.predict(X_train)


# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_train_bin = np.where(y_pred_train == -1, 0, 1)

# Evaluate the One-Class SVM model on the training set
print("Accuracy on the training set using One-Class SVM:")
print(accuracy_score(y_train, y_pred_train_bin))
print("\nClassification report for One-Class SVM on training set:")
print(classification_report(y_train, y_pred_train_bin))

