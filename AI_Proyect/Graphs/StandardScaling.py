import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Upload the data
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop irrelevant columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Drop duplicates
df = df.drop_duplicates()

# Identify and transform categorical columns
df['Dport'] = df['Dport'].astype('object')
categorical_columns = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Apply StandardScaler to all columns
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Create the graphs of the distribution of each column
for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=df[column], inner='quartile')
    plt.title(f'Distribución de {column}')
    plt.show()