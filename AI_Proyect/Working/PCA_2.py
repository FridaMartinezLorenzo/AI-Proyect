import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

df = df.drop(['Dir', 'Flgs'], axis=1)

df = df.drop_duplicates()

df['Dport'] = df['Dport'].astype('object')
categorical_columns = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Apply the Min-Max Scaling to all the columns
#scaler = MinMaxScaler()
#df[df.columns] = scaler.fit_transform(df[df.columns])

# Apply StandardScaler to all the columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Apply PCA
pca = PCA()

# Select the number of principal components to explain the 95% of variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Number of the principal components to explain the 95% of variance: {n_components}')

# Inicialize the PCA with the number of selected components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df)

# Create a new Dataframe
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Show the new Dataframe
print(df_pca)

# Save the new dataset
df_pca.to_csv('dataset_PCA.csv', index=False)

# Show details of the results of the PCA
print("Principal Components (Eigenvectors):\n", pca.components_)
print("\nVariance (Eigenvalues):\n", pca.explained_variance_)
print("\nProportion of Variance:\n", pca.explained_variance_ratio_)
