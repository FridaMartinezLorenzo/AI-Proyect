import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

# Load the training and testing datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Split the data into features and labels
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid for search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model with the best combination of hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameter combination
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Evaluate the model with the test data using the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate AUC for each class and then average
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
if y_test_binarized.shape[1] == y_pred_proba.shape[1]:
    auc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
else:
    auc = None  # AUC is not applicable for binary or single-class scenarios

print(f"Model accuracy with optimized hyperparameters: {accuracy}")
print(f"Model F1-score with optimized hyperparameters: {f1}")
print(f"Model precision with optimized hyperparameters: {precision}")
print(f"Model recall with optimized hyperparameters: {recall}")
if auc is not None:
    print(f"Model AUC with optimized hyperparameters: {auc}")
else:
    print("AUC is not applicable for this dataset.")
