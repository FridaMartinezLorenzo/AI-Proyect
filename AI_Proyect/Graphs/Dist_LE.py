import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

# Aplicar LabelEncoder en columnas categóricas
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Seleccionar solo columnas numéricas para calcular la correlación
df_numeric = df.select_dtypes(include=[np.number])

# Configurar el tamaño de fuente general
sns.set(font_scale=0.7)  # Ajusta el factor de escala

# Matriz de correlación después del Label Encoding
plt.figure(figsize=(10, 8))
plt.title("Matriz de correlación después del Label Encoding", fontsize=10)
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
#Save the graph
plt.savefig('Dist_LE.png')

plt.tight_layout()
plt.show()
