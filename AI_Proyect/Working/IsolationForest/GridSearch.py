import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']
# Feature Selection using Random Forest (could also use other models)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Select important features
selector = SelectFromModel(rf, threshold='median')
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Adjusting the GridSearchCV with the selected features
param_grid_refined = {
    'n_estimators': [200, 300, 400],
    'max_samples': [0.5, 0.75, 1.0],
    'contamination': [0.1, 0.12, 0.15],
    'max_features': [0.5, 0.75, 1.0],
    'bootstrap': [True, False]
}

iso_forest_refined = IsolationForest(random_state=42)
grid_search_refined = GridSearchCV(estimator=iso_forest_refined, param_grid=param_grid_refined, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search_refined.fit(X_train_selected, y_train)

best_params_refined = grid_search_refined.best_params_
best_model_refined = grid_search_refined.best_estimator_

y_pred_test_refined = best_model_refined.predict(X_test_selected)
y_pred_test_bin_refined = np.where(y_pred_test_refined == -1, 0, 1)

print("Best parameters found by refined Grid Search:")
print(best_params_refined)

print("\nAccuracy on the test set using the refined Isolation Forest model:")
print(accuracy_score(y_test, y_pred_test_bin_refined))

print("\nClassification report for the refined Isolation Forest model on test set:")
print(classification_report(y_test, y_pred_test_bin_refined))