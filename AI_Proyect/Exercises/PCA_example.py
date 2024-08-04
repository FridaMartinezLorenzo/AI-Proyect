import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Paso 1: Cargar el dataset
# Supongamos que tenemos un dataset en un archivo CSV
df = pd.read_csv('ruta_al_archivo/dataset.csv')

# Paso 2: Estandarizar los datos
# Separar las características y la variable objetivo si es necesario
X = df.drop('variable_objetivo', axis=1)
y = df['variable_objetivo']

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Paso 3: Aplicar PCA
# Inicializar PCA
pca = PCA()

# Ajustar PCA a los datos
pca.fit(X_scaled)

# Paso 4: Seleccionar el número de componentes principales
# Supongamos que queremos explicar el 95% de la varianza
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Número de componentes principales para explicar el 95% de la varianza: {n_components}')

# Inicializar PCA con el número de componentes seleccionados
pca = PCA(n_components=n_components)

# Ajustar y transformar los datos
X_pca = pca.fit_transform(X_scaled)

# Paso 5: Crear un nuevo DataFrame con las componentes principales
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
df_pca['variable_objetivo'] = y.values

# Paso 6: Guardar el nuevo dataset en un archivo CSV
df_pca.to_csv('ruta_al_archivo/nuevo_dataset.csv', index=False)

print('Nuevo dataset guardado con éxito.')
