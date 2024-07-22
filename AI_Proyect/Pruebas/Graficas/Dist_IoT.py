import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = '../WUSTL-EHMS/wustl-ehms-2020.csv'
df = pd.read_csv(file_path)

# Atributos relacionados con dispositivos-aplicación en IoT
iot_columns = ['SrcAddr', 'DstAddr', 'Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
total_columns = len(df.columns)
columns_no_iot = total_columns - len(iot_columns)
columns_iot = len(iot_columns)

# Graficar cuantos atributos pertenecen a IoT y cuantos no
plt.figure(figsize=(8, 6))
labels = ['Non-IoT attributes', 'IoT attributes']
counts = [columns_no_iot, columns_iot]
colors = ['pink', 'magenta']

plt.bar(labels, counts, color=colors)
plt.title('Distribution of dispositives-app IoT')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

## Graficar la distribución de los dispositivos-aplicación en IoT una por una
#for column in iot_columns:
#    plt.figure(figsize=(10, 6))
#    if pd.api.types.is_numeric_dtype(df[column]):
#        df[column].plot(kind='hist', bins=30, color='orange', edgecolor='black')
#    else:
#        df[column].value_counts().head(20).plot(kind='bar', color='orange')
#    plt.title(f'Distribution of {column}')
#    plt.xlabel(column)
#    plt.ylabel('Count')
#    plt.grid(axis='y')
#    plt.tight_layout()
#    plt.show()
#
