import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

# Load the dataset
EHMS = pd.read_csv('../../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop unnecessary columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Convert 'Dport' to an object type
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

# Apply StandardScaler to all the columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Apply PCA
pca = PCA()
pca.fit(df)

# Select the number of principal components to explain 95% of the variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Number of principal components to explain 95% of the variance: {n_components}')

# Initialize PCA with the selected number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df)

# Create a new DataFrame with the principal components
feature_names = [f'PC_{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(X_pca, columns=feature_names)

# Add the label column back to the PCA-transformed DataFrame
df_pca = pd.concat([df_pca, label_column.reset_index(drop=True)], axis=1)

print(df_pca)

# Define features and target
X = df_pca.drop(columns=['Label'])
y = df_pca['Label']

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize KFold cross-validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# Train the Random Forest model on the entire dataset
rf.fit(X, y)

