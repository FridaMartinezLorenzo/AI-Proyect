import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
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

# Display the new DataFrame
print(df_pca)

# Save the new dataset
df_pca.to_csv('dataset_PCA.csv', index=False)

# Display details of the PCA output
print("Principal Components (Eigenvectors):\n", pca.components_)
print("\nVariance (Eigenvalues):\n", pca.explained_variance_)
print("\nProportion of Variance:\n", pca.explained_variance_ratio_)

# Map the principal components back to the original feature names
loading_scores = pd.DataFrame(pca.components_.T, index=df.columns, columns=feature_names)
print("\nLoading Scores:\n", loading_scores)

# List to store the names of the main features
features = []

# Identify which original features contribute the most to each principal component
for pc in loading_scores.columns:
    print(f'\nTop contributing features to {pc}:')
    top_features = loading_scores[pc].nlargest(10).index.tolist()
    print(top_features)
    # Add the main feature names to the 'features' list
    features.extend(top_features)

# Convert to set to remove duplicates and then back to list
unique_features = list(set(features))
print("\nTop contributing features of each PC (without duplicates):\n", unique_features)

# Count how many times each feature appears in the list of main features
feature_counts = {feature: features.count(feature) for feature in unique_features}

# Order the features based on their incidence number
feature_counts = dict(sorted(feature_counts.items(), key=lambda item: item[1], reverse=True))

print("\nNumber of unique main features:", len(unique_features))
print("\nFrequency of each main feature:\n", feature_counts)

# We are going to work with these new features, so we are going to redo the dataset with only these features
df_selected_features = df[unique_features]

# Add the label column
df_selected_features = pd.concat([df_selected_features, label_column], axis=1)

# Save the new dataset with selected features
df_selected_features.to_csv('dataset_selected_features.csv', index=False)
print("\nDataset with selected features saved.")

