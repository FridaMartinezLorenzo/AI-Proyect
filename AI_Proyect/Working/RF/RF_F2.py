import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Load the dataset
EHMS = pd.read_csv('../../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop unnecessary columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Convert 'Dport' to object type
df['Dport'] = df['Dport'].astype('object')

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create a copy of the 'Label' column
label_column = df[['Label']]
df = df.drop(columns=["Label"])


# Scale the features
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
# Split the label and features
X = df
y = label_column.values.ravel()  # Flatten the array

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Save feature importances to a CSV file
try:
    feature_importances_df.to_csv('feature_importances2_df.csv', index=False)
    print("Feature importances saved successfully to 'feature_importances_df.csv'")
except Exception as e:
    print(f"Error saving feature importances: {e}")

# Select the most important features
threshold = 0.01  # Importance threshold
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filter the dataset with the selected features
X_important = df[important_features]
X_important = pd.concat([X_important, label_column], axis=1)

#print(X_important)
#Considering the biomedical data, we are going  to count how many features are biomedical
biometric_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate','ST']
number_of_biometric_features = len(X_important.columns.intersection(biometric_columns))

print("\n______________________________________________________\n Counting the number of biometric features\n______________________________________________________")
print("\nNumber of biometric features:", number_of_biometric_features)
print("Number of network flow features:", len(X_important.columns) - number_of_biometric_features)


# Save the filtered dataset to a new CSV file
try:
    X_important.to_csv('filtered_dataset2.csv', index=False)
    print("Filtered dataset saved successfully to 'filtered_dataset.csv'")
except Exception as e:
    print(f"Error saving filtered dataset: {e}")

# Further steps for training and evaluating the model using selected features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

print("X_train_important")
print(X_train_important)
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
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
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
