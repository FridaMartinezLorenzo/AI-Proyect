import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load dataset
EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Delete irrelevant columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Delete duplicates
df = df.drop_duplicates()

# Identify the categorical columns to apply LabelEncoder
df['Dport'] = df['Dport'].astype('object')

#Consider only specific columns
columns_to_consider =['Sport', 'SrcLoad','DstLoad','SIntPkt','DIntPkt','SrcJitter','DstJitter','Dur','Load','Rate','SrcMac','Packet_num', 'Temp','SpO2','Pulse_Rate','SYS','DIA','Heart_rate','Resp_Rate','ST','Label']
df = df[columns_to_consider]


categorical_columns = df.select_dtypes(include=['object']).columns

# Separate the label column
label_column = df['Label']
df = df.drop(['Label'], axis=1)

# Apply LabelEncoder to the categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Apply StandardScaler to all columns
#scaler = StandardScaler()
#df[df.columns] = scaler.fit_transform(df[df.columns])

#Apply MinMaxScaler to all columns
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Combine the features and labels
df_with_label = pd.concat([df, label_column], axis=1)

# Separate features (X) and labels (y) again for SMOTE
X = df_with_label.drop(columns=['Label'])
y = df_with_label['Label']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and labels
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Label')], axis=1)

# Show the dataframe after SMOTE and scaling
print(df_resampled.head())

# Save the balanced and preprocessed dataset
#df_resampled.to_csv('dataset_pre_processed_standard_balanced.csv', index=False)
df_resampled.to_csv('dataset_pre_processed_minmax_balanced.csv', index=False)