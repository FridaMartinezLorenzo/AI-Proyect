import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Load the training and testing datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Split the data into features and labels
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Convert y to a one-dimensional array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Get the feature names
feature_names = X_train.columns

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=166,
                            max_depth=10,
                            min_samples_split=9,
                            min_samples_leaf=3,
                            max_features='log2',
                            bootstrap=True,
                            random_state=42)
rf.fit(X_train, y_train)

print(f'Training accuracy: {rf.score(X_train, y_train)}')

# Filter the dataset with the valid features using Pandas DataFrame selection

# Get the feature importances
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Select the most important features
threshold = 0.01  # Importance threshold
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filter the dataset with the valid features using Pandas DataFrame selection
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

#_______________________________________________________________________________
# Train the Random Forest model again with the important features
rf_important = RandomForestClassifier(n_estimators=166,
                                      max_depth=10,
                                      min_samples_split=9,
                                      min_samples_leaf=3,
                                      max_features='log2',
                                      bootstrap=True,
                                      random_state=42)
rf_important.fit(X_train_important, y_train)

#_______________________________________________________________________________
# Compute SHAP values
try:
    explainer = shap.TreeExplainer(rf_important)
except Exception as e:
    print(f"Error creating SHAP explainer: {e}")
    
class_names = np.unique(y_test)

shap_values = explainer.shap_values(X_test_important)
print("Test features :", X_test_important)

# If shap_values is a list (multi-class), sum across classes first
if isinstance(shap_values, list):
    shap_values_stacked = np.sum(np.abs(shap_values), axis=0)  # Sum absolute SHAP values across classes
else:
    shap_values_stacked = np.abs(shap_values)  # Take absolute SHAP values if binary classification

# Now, take the mean across samples
shap_importance = shap_values_stacked.mean(axis=0)

# If shap_importance is still 2D (features x classes), sum across the class axis
if shap_importance.ndim == 2:
    shap_importance = shap_importance.sum(axis=1)

# Check the shape of shap_importance to confirm it is 1D
print(f'Shape of shap_importance: {shap_importance.shape}')  # Should be (number of features,)

# Create a DataFrame for the SHAP feature importances
shap_importance_df = pd.DataFrame({
    'Feature': X_test_important.columns,
    'SHAP Importance': shap_importance
}).sort_values(by='SHAP Importance', ascending=False)

# Plotting the SHAP feature importance as a bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x='SHAP Importance', y='Feature', data=shap_importance_df)
plt.title("SHAP Feature Importance")
plt.show()