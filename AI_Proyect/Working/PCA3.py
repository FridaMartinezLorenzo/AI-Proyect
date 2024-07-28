import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

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

# Calculate the mean
print('----------------------')
print('Mean of each variable')
print('----------------------')
print(df.mean(axis=0))

# Calculate the variance
print('-------------------------')
print('Variance of each variable')
print('-------------------------')
print(df.var(axis=0))

# Apply StandardScaler to all the columns
df_aux = df.copy()
scaler = StandardScaler()
df_aux[df_aux.columns] = scaler.fit_transform(df_aux[df_aux.columns])

pca = PCA()
pca.fit(df_aux)

# Select the number of principal components to explain 95% of the variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_v = np.argmax(explained_variance >= 0.95) + 1
print(n_components_v)

# Training PCA model with data scaling
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=n_components_v))
X_pca = pca_pipe.fit_transform(df)

# Extract the trained PCA model from the pipeline
modelo_pca = pca_pipe.named_steps['pca']

# Convert the array to a DataFrame to add names to the axes
feature_names = [f'PC_{i+1}' for i in range(modelo_pca.n_components_)]
df_pca_components = pd.DataFrame(
    data=modelo_pca.components_,
    columns=df.columns,
    index=feature_names
)

# Print the new DataFrame with principal components
print(df_pca_components)

# Save the new dataset
df_pca_components.to_csv('dataset_PCA.csv', index=False)

# Heatmap components
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
components = modelo_pca.components_
plt.imshow(components.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(df.columns)), df.columns)
plt.xticks(range(modelo_pca.n_components_), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.title('Heatmap of PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Original Features')
plt.show()

# Explained variance ratio by each component
print('----------------------------------------------------')
print('Explained variance ratio by each component')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x=np.arange(modelo_pca.n_components_) + 1,
    height=modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(modelo_pca.n_components_) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Explained variance ratio by each component')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
plt.show()

# Cumulative explained variance ratio
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Cumulative explained variance ratio')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(modelo_pca.n_components_) + 1,
    prop_varianza_acum,
    marker='o'
)

for x, y in zip(np.arange(modelo_pca.n_components_) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )

ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Cumulative explained variance ratio')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Projection of training observations
projections = pd.DataFrame(
    X_pca,
    columns=feature_names
)

print("Projections Head\n", projections.head())

# Reconstruction of projections
reconstruction = pca_pipe.inverse_transform(X_pca)
reconstruction = pd.DataFrame(
    reconstruction,
    columns=df.columns,
    index=df.index
)

#Adding the original label column
reconstruction['Label'] = label_column
df['Label'] = label_column

print('------------------')
print('Original values')
print('------------------')
print(reconstruction.head())

print('---------------------')
print('Reconstructed values')
print('---------------------')
print(df)
