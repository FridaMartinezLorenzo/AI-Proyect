import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Load the dataset
EHMS = pd.read_csv('dataset_PCA_MINMAX.csv')
df = pd.DataFrame(EHMS)

# Define features and target
X = df.drop(columns=['Label'])
y = df['Label']

#Loading the test and train datasets
X_train = pd.read_csv('../TrainTest/Split/X_train.csv')

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize KFold cross-validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model on the training set
rf.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

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
