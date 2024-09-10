# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle   

# Load the dataset
EHMS = pd.read_csv('../../Wustl-EHMS dataset/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Define the columns to consider
columns_to_consider = ['DstBytes','Dport','DstMac','DstAddr','SrcAddr','SrcBytes','Sport', 'SrcLoad', 'DstLoad', 'SIntPkt', 'DIntPkt', 'SrcJitter', 'DstJitter', 'Dur', 'Load', 'Rate', 'SrcMac', 
                       'Packet_num', 'Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST','dMaxPktSz','dMinPktSz','Loss','pLoss','pSrcLoss','TotPkts']


# Filter the dataset with the specified columns
df_filtered = df[columns_to_consider]


# Create a vocabulary based on the values of each column of the dataset
def create_vocabulary(df):
    vocabulary = []
    for column in df.columns:
        vocabulary.extend(df[column].unique())
    return vocabulary

vocabulary = create_vocabulary(df_filtered)


def OneHotEncoding_vector(element, vocabulary):
    vector = [0] * len(vocabulary)
    vector[vocabulary.index(element)] = 1
    return vector

# read the rows of the dataset and read each element
def read_dataset_elements(df,vocabulary):
    for index, row in df.iterrows():
        oneHotencoding_vectors=[]
        for column in df.columns:
            element = row[column]
            vector=OneHotEncoding_vector(element, vocabulary)
            oneHotencoding_vectors.append(vector)
            # save in pkl file
            with open(f'/home/mlhj/HDD512/FridaProject/OneHotEncoded_vectors/row_{index}.pkl', 'wb') as f:
                pickle.dump(oneHotencoding_vectors, f)



# Call the function to read the dataset elements
read_dataset_elements(df_filtered,vocabulary)