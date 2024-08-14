import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from scipy.stats import randint, uniform

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

# Define the hyperparameter grid for random search
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Configure the random search with cross-validation
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_distributions,
                                   n_iter=100,  # Number of hyperparameter combinations to try
                                   cv=5,  # Number of folds for cross-validation
                                   n_jobs=-1,  # Use all available cores
                                   verbose=2,
                                   random_state=42,  # Ensure reproducibility
                                   scoring='accuracy')

# Train the model with the best combination of hyperparameters
random_search.fit(X_train, y_train)

# Get the best hyperparameter combination
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Evaluate the model with the test data using the best model
best_rf = random_search.best_estimator_
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
