import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

# Verificar columnas categóricas
print("Columnas categóricas antes de One-Hot Encoding:", categorical_columns)

# Aplicar One-Hot Encoding en columnas categóricas
df = pd.get_dummies(df, columns=categorical_columns)

# Convertir columnas booleanas (0, 1) a tipo int
df = df.astype(int)

# Verificar la conversión
print("Columnas después de One-Hot Encoding:", df.columns)

# Verificar el número de columnas
print("Número de columnas después de One-Hot Encoding:", len(df.columns))

# Seleccionar solo columnas numéricas para calcular la correlación
df_numeric = df.select_dtypes(include=[np.number])

print("Número de columnas numéricas después del One-Hot Encoding:", len(df_numeric.columns))

# Configurar el tamaño de fuente general
sns.set(font_scale=0.7)

# Matriz de correlación antes del One-Hot Encoding
df_original_numeric = df_original.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
plt.title("Matriz de correlación antes del One-Hot Encoding", fontsize=10)
sns.heatmap(df_original_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_no_OHE.png')
plt.show()

# Matriz de correlación después del One-Hot Encoding
plt.figure(figsize=(10, 8))
plt.title("Matriz de correlación después del One-Hot Encoding", fontsize=10)
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_OHE.png')
plt.show()


