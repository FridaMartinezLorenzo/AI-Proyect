import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')

df = pd.DataFrame(EHMS)

# Convertir la columna 'Label' a categórica
df['Dport'] = df['Dport'].astype('category')

# Análisis de cada columna
column_analysis = []

for column in df.columns:
    col_type = df[column].dtype

    if pd.api.types.is_numeric_dtype(df[column]):
        col_min = df[column].min()
        col_max = df[column].max()
        column_analysis.append({
            'Column': column,
            'DataType': col_type,
            'Category': 'Numeric',
            'Min': col_min,
            'Max': col_max
        })
    else:
        column_analysis.append({
            'Column': column,
            'DataType': col_type,
            'Category': 'Categorical',
            'Min': None,
            'Max': None
        })

# Convertir el análisis a un DataFrame para una visualización más fácil
analysis_df = pd.DataFrame(column_analysis)

# Mostrar el análisis
print(analysis_df)


#vamos a graficar la distribución de los datos biometricos
biometric_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate','ST']


#Cuidados en un ambiente hospitalario, para evitar el robo de información médica o la alteración de la misma

for column in biometric_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=20, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.show()

#flow_metric_colums = ['SrcBytes', 'DstBytes', 'SrcGap', 'DstGap','SrcJitter','DstJitter','sMaxPktSz','dMaxPktSz','sMinPktSz','dMinPktSz','Dur','Trans','TotPkts','TotBytes']
#for column in flow_metric_colums:
#    plt.figure(figsize=(8, 6))
#    plt.hist(df[column], bins=20, color='pink')
#    plt.title(f'Distribution of {column}')
#    plt.xlabel(column)
#    plt.ylabel('Count')
#    plt.grid(axis='y')
#    plt.show()


#df_graph = df.copy()
#for column in biometric_columns:
#    df[column] = df[column].apply(lambda x: set(x))
#    #Vamos a hacer una grafica de lineas y no de barras para que se aprecien la distribucion de los datos
#    df_graph[column].plot(kind='line')
#    plt.title(column)
#    plt.show()    
    
