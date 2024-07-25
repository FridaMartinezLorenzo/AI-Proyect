import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.decomposition import PCA

EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

#We delete the irrelevant columns, in this case
df = df.drop(['Dir', 'Flgs'], axis=1)

# Delete duplicates
df = df.drop_duplicates()

#Identify the categorical columns to aply the LabelEncoder
#We change manually this attribute ‘cause the system did not detect it as categorical
df['Dport'] = df['Dport'].astype('object')
categorical_columns = df.select_dtypes(include=['object']).columns

# We aply the LabelEncoder in the cathegorical_columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform( ( df[col] ))
    #print("Attribute",col, "Classes:", len(list(le.classes_)))
    label_encoders[col] = le



# Aply the StandardScaler to all the columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Aplicar PCA
pca = PCA()
pca.fit(df)

# Seleccionar el número de componentes principales para explicar el 95% de la varianza
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Número de componentes principales para explicar el 95% de la varianza: {n_components}')

# Inicializar PCA con el número de componentes seleccionados
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(df)

# Crear un nuevo DataFrame con las componentes principales
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# Si existe una variable objetivo, puedes agregarla de nuevo
# Supongamos que tienes una columna 'Label' en el dataset original
# df_pca['Label'] = EHMS['Label'].values

# Mostrar el DataFrame resultante
print(df_pca)

# Guardar el nuevo dataset en un archivo CSV
df_pca.to_csv('dataset_PCA.csv', index=False)