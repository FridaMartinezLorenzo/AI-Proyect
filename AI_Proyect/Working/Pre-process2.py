import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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

# Separate label column
label_column = df['Label']
df = df.drop(['Label'], axis=1)

# Apply One-Hot Encoding on categorical columns
df = pd.get_dummies(df, columns=categorical_columns)

# Convert boolean columns (0, 1) to int type
df = df.astype(int)

# Add the label column back
df = pd.concat([df, label_column], axis=1)

# Separate features and label
X = df.drop(columns=['Label'])
y = df['Label']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify the distribution of the classes after SMOTE
print("Distribution of classes after SMOTE:")
print(y_resampled.value_counts())

# Apply StandardScaler to the resampled dataset
scaler = StandardScaler()
X_resampled[X_resampled.columns] = scaler.fit_transform(X_resampled[X_resampled.columns])

#Apply Min-Max Scaling to the resampled dataset
#scaler = MinMaxScaler()
#X_resampled[X_resampled.columns] = scaler.fit_transform(X_resampled[X_resampled.columns])

# Combine the resampled and scaled features with the labels
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Show the dataframe after SMOTE and scaling
print(df_resampled.head())

# Save the new dataset
#df_resampled.to_csv('dataset_pre_processed_balanced_minmax.csv', index=False)
df_resampled.to_csv('dataset_pre_processed_balanced_standard.csv', index=False)
