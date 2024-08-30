# This code was adapted from an example in the scikit-learn documentation.
# We modified the code to suit our specific needs, evaluating the model 
# using cross-validation and selecting the best hyperparameters.

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

# Define the custom refit strategy function
def refit_strategy(cv_results):
    """Strategy to select the best estimator based on precision and recall."""
    precision_threshold = 0.98

    # Convert the cv_results dictionary to a DataFrame
    cv_results_ = pd.DataFrame(cv_results)
    
    # Filter out all results below the precision threshold
    high_precision_cv_results = cv_results_[cv_results_["mean_test_precision"] > precision_threshold]
    
    if high_precision_cv_results.empty:
        print(f"No models met the precision threshold of {precision_threshold}.")
        return cv_results_["mean_test_precision"].idxmax()  # Select the model with the highest precision

    # Select models within one standard deviation of the best recall
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std
    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    
    if high_recall_cv_results.empty:
        print("No models met the recall criteria, selecting the best precision model instead.")
        return high_precision_cv_results["mean_test_precision"].idxmax()  # Select the model with the highest precision

    # Select the fastest model to predict among the shortlisted models
    fastest_top_recall_high_precision_index = high_recall_cv_results["mean_score_time"].idxmin()
    
    return fastest_top_recall_high_precision_index


# Define the scoring metrics
scores = ["precision", "recall"]

# Set up the grid search with cross-validation using the custom refit strategy
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring=scores, refit=refit_strategy,
                           cv=5, n_jobs=-1, verbose=2)

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
