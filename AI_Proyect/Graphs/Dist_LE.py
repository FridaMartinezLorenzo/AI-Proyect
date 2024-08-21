import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

# Cargar datos
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Eliminar columnas irrelevantes y duplicados
df = df.drop(['Dir', 'Flgs'], axis=1)
df = df.drop_duplicates()

# Identificar columnas categóricas
df['Dport'] = df['Dport'].astype('object')
categorical_columns = df.select_dtypes(include=['object']).columns

df_original = df.copy()

label_column = df['Label']
df = df.drop(['Label'], axis=1)

# Aplicar LabelEncoder en columnas categóricas
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# Seleccionar solo columnas numéricas para calcular la correlación
df_numeric = df.select_dtypes(include=[np.number])

# Configurar el tamaño de fuente general
sns.set(font_scale=0.7)

# Matriz de correlación antes del Label Encoding
df_original_numeric = df_original.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
plt.title("Matriz de correlación antes del Label Encoding", fontsize=10)
sns.heatmap(df_original_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_no_LE.png')
plt.show()

# Matriz de correlación después del Label Encoding
plt.figure(figsize=(10, 8))
plt.title("Matriz de correlación después del Label Encoding", fontsize=10)
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_LE.png')
plt.show()

"""
Aplicar la normalización a todas las columnas
"""

# Aplicar StandardScaler a todas las columnas
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Agregar la columna de etiquetas
df = pd.concat([df, label_column], axis=1)

# Graficar la distribución de las columnas antes y después de la normalización
for column in df.columns[:-1]:  # Excluye la columna de etiquetas
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma de la columna original
    sns.histplot(data=df_original, x=column, kde=True, ax=ax[0])
    ax[0].title.set_text(f'Distribución Original de {column}')

    # Histograma de la columna normalizada
    sns.histplot(data=df, x=column, kde=True, ax=ax[1])
    ax[1].title.set_text(f'Distribución Normalizada de {column}')

    plt.savefig(f'Normalization/Dist_{column}.png')
    #plt.show()