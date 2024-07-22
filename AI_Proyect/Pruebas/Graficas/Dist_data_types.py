import pandas as pd
import matplotlib.pyplot as plt

EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')

df = pd.DataFrame(EHMS) 
#print(EHMS.head())

# Convertir tipos de datos si es necesario para categorizar, lo estaba detectando como numerico
df['Dport'] = df['Dport'].astype('object')

# Identificar las características categóricas y numéricas
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Contar las características categóricas y numéricas
categorical_count = len(categorical_columns)
numerical_count = len(numerical_columns)

# Graficar la distribución de las características por formato
plt.figure(figsize=(8, 6))
plt.bar(['Categorical', 'Numerical'], [categorical_count, numerical_count], color=['Orange', 'Pink'])
plt.title('Distribution by Category')
plt.xlabel('Feature Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

categorical_columns, numerical_columns
