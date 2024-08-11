import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Load the dataset
EHMS = pd.read_csv('../dataset_pre_processed_minmax.csv')
df = pd.DataFrame(EHMS)

# Split the dataset into features and labels
X = df.drop(columns=['Label'])
y = df[['Label']]

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

# Add the label column
df_pca = pd.concat([df_pca, y], axis=1)

# Save the new dataset
df_pca.to_csv('dataset_PCA_MINMAX.csv', index=False)

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

#Considering the biomedical data, we are going  to count how many features are biomedical
biometric_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate','ST']
number_of_biometric_features = len(df_selected_features.columns.intersection(biometric_columns))

print("\n______________________________________________________\n Counting the number of biometric features\n______________________________________________________")
print("\nNumber of biometric features:", number_of_biometric_features)
print("Number of network flow features:", len(df_selected_features.columns) - number_of_biometric_features)

# Add the label column
df_selected_features = pd.concat([df_selected_features, y], axis=1)

# Save the new dataset with selected features
df_selected_features.to_csv('dataset_selected_features_MINMAX.csv', index=False)
print("\nDataset with selected features saved.")



