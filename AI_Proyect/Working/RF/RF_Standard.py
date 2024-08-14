import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

# Cargar los datasets de entrenamiento y prueba
train_data = pd.read_csv('../TrainTest/Split/train_Standard.csv')
test_data = pd.read_csv('../TrainTest/Split/test_Standard.csv')

# Split the data into features and labels
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Handle missing values in the features
# Impute missing values in features with the mean
imputer_X = SimpleImputer(strategy='mean')
X_train_imputed = imputer_X.fit_transform(X_train)
X_test_imputed = imputer_X.transform(X_test)

# Handle missing values in the labels
# Option 1: Remove rows with missing labels
non_nan_train_indices = ~y_train.isna()
non_nan_test_indices = ~y_test.isna()

X_train_imputed = X_train_imputed[non_nan_train_indices]
y_train = y_train[non_nan_train_indices]

X_test_imputed = X_test_imputed[non_nan_test_indices]
y_test = y_test[non_nan_test_indices]

# Convert y to a one-dimensional array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Get the feature names
feature_names = X_train.columns

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_imputed, y_train)

# Get the feature importances
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Save the feature importances to a CSV file
try:
    feature_importances_df.to_csv('feature_importances1_df.csv', index=False)
    print("Feature importances saved successfully to 'feature_importances1_df.csv'")
except Exception as e:
    print(f"Error saving feature importances: {e}")

# Select the most important features
threshold = 0.01  # Importance threshold
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filter only the features that exist in the training data
valid_important_features = [feature for feature in important_features if feature in feature_names]
print("Valid features:", valid_important_features)

# Filter the dataset with the valid features
X_train_important = X_train_imputed[:, [list(feature_names).index(feature) for feature in valid_important_features]]
X_test_important = X_test_imputed[:, [list(feature_names).index(feature) for feature in valid_important_features]]

# Count the number of biometric features
biometric_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
number_of_biometric_features = len(np.intersect1d(valid_important_features, biometric_columns))

print("\n______________________________________________________")
print("Counting the number of biometric features")
print("______________________________________________________")
print("\nNumber of biometric features:", number_of_biometric_features)
print("Number of network flow features:", len(valid_important_features) - number_of_biometric_features)

# Save the filtered dataset to a new CSV file
X_train_important_df = pd.DataFrame(X_train_important, columns=valid_important_features)
X_test_important_df = pd.DataFrame(X_test_important, columns=valid_important_features)

try:
    X_train_important_df.to_csv('train_filtered_dataset.csv', index=False)
    X_test_important_df.to_csv('test_filtered_dataset.csv', index=False)
    print("\nFiltered datasets saved successfully to 'train_filtered_dataset.csv' and 'test_filtered_dataset.csv'")
except Exception as e:
    print(f"\nError saving filtered datasets: {e}")

# Train a new model with the selected features
rf_important = RandomForestClassifier(n_estimators=100, random_state=42)
rf_important.fit(X_train_important, y_train)

# Predict and evaluate the model
y_pred = rf_important.predict(X_test_important)
y_pred_proba = rf_important.predict_proba(X_test_important)

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

print(f"Model accuracy with selected features: {accuracy}")
print(f"Model F1-score with selected features: {f1}")
print(f"Model precision with selected features: {precision}")
print(f"Model recall with selected features: {recall}")
if auc is not None:
    print(f"Model AUC with selected features: {auc}")
else:
    print("AUC is not applicable for this dataset.")
