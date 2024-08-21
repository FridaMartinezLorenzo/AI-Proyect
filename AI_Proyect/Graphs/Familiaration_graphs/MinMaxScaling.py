import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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


# Aply the Min-Max Scaling to all the columns
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])


#Show the dataframe
#print(df.head())
#print(df.tail())
print(df)


# Create the graphs of the distribution of each column
for column in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=df[column], inner='quartile')
    plt.title(f'Distribución de {column}')
    plt.show()