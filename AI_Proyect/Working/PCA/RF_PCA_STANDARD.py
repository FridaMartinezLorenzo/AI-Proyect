import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_Standard.csv')
test_data = pd.read_csv('../TrainTest/Split/test_Standard.csv')

# Extract features and labels from train and test datasets
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Impute missing values in the features with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Handle missing values in the target labels
# Option 1: Drop rows with missing labels
non_nan_train_indices = ~y_train.isna()
non_nan_test_indices = ~y_test.isna()

X_train = X_train[non_nan_train_indices]
y_train = y_train[non_nan_train_indices]

X_test = X_test[non_nan_test_indices]
y_test = y_test[non_nan_test_indices]

# Apply PCA to the training data to determine the number of components
pca = PCA()
pca.fit(X_train)

# Select the number of principal components to explain 95% of the variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Number of principal components to explain 95% of the variance: {n_components}')

# Initialize PCA with the selected number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model on the training set
rf.fit(X_train_pca, y_train)

# Predict on the testing set
y_pred = rf.predict(X_test_pca)
y_prob = rf.predict_proba(X_test_pca)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy on the testing set: {accuracy}')
print(f'F1-score on the testing set: {f1}')
print(f'AUC score on the testing set: {auc}')
print(f'Precision on the testing set: {precision}')
print(f'Recall on the testing set: {recall}')
