# Import necessary libraries
import pandas as pd

# Load the uploaded dataset
EHMS = pd.read_csv('../../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Check the columns in the DataFrame
print("Columnas del DataFrame:", df.columns.tolist())

# Define the columns to consider
columns_to_consider = ['Sport', 'SrcLoad', 'DstLoad', 'SIntPkt', 'DIntPkt', 'SrcJitter', 'DstJitter', 'Dur', 'Load', 
                       'Rate', 'SrcMac', 'Packet_num', 'Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 
                       'Resp_Rate', 'ST', 'Label']

# Filter the dataset with the specified columns
df_filtered = df

# Check the first few rows to ensure proper filtering
print("Primeras filas del DataFrame filtrado:", df_filtered.head())

# Create a dictionary to represent the vocabulary for each category
connections_vocabulary = {
    "DstBytes": df_filtered['DstLoad'].unique(),
    "SrcBytes": df_filtered['SrcLoad'].unique(),
    "DstAddr": df_filtered['DstMac'].unique() if 'DstMac' in df_filtered else None,
    "SrcAddr": df_filtered['SrcMac'].unique() if 'SrcMac' in df_filtered else None,
    "Sport": df_filtered['Sport'].unique(),
    "Dport": df_filtered['Sport'].unique(), 
    "SrcMac": (df_filtered['SrcMac'].iloc[0] if df_filtered['SrcMac'].nunique() == 1 else df_filtered['SrcMac'].unique()),
    "DstMac": (df_filtered['DstMac'].iloc[0] if df_filtered['DstMac'].nunique() == 1 else df_filtered['DstMac'].unique())
}

# Print the connections vocabulary
print("Vocabulario de conexiones:", connections_vocabulary)
