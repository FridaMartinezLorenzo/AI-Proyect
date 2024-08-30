import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

# Load data
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop irrelevant columns and duplicates
df = df.drop(['Dir', 'Flgs'], axis=1)
df = df.drop_duplicates()

# Identify categorical columns
df['Dport'] = df['Dport'].astype('object')
categorical_columns = df.select_dtypes(include=['object']).columns

df_original = df.copy()

label_column = df['Label']
df = df.drop(['Label'], axis=1)

print("Categorical columns before Label Encoding:", df)

# Apply LabelEncoder on categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Columns after Label Encoding:", df)

# Select only numeric columns to calculate the correlation
df_numeric = df.select_dtypes(include=[np.number])

# Set the general font size
sns.set(font_scale=0.7)

# Correlation matrix before Label Encoding
df_original_numeric = df_original.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix before Label Encoding", fontsize=10)
sns.heatmap(df_original_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_no_LE.png')
plt.show()

# Correlation matrix after Label Encoding
print("Number of numeric columns:", len(df_numeric.columns))
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix after Label Encoding", fontsize=10)
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_LE.png')
plt.show()

"""
Apply normalization to all columns

# Apply StandardScaler to all columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Add the label column
df = pd.concat([df, label_column], axis=1)

"""
# Graficar la distribución de las columnas antes y después de la normalización
#for column in df.columns[:-1]:  # Excluye la columna de etiquetas
#    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#
#    # Histograma de la columna original
#    sns.histplot(data=df_original, x=column, kde=True, ax=ax[0])
#    ax[0].title.set_text(f'Distribución Original de {column}')
#
#    # Histograma de la columna normalizada
#    sns.histplot(data=df, x=column, kde=True, ax=ax[1])
#    ax[1].title.set_text(f'Distribución Normalizada de {column}')
#
#    plt.savefig(f'Normalization/Dist_{column}.png')
#    #plt.show()