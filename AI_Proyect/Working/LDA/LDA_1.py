import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the dataset
EHMS = pd.read_csv('../../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop unnecessary columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Convert 'Dport' to object type
df['Dport'] = df['Dport'].astype('object')

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create a copy of the 'Label' column
label_column = df[['Label']]
df = df.drop(columns=["Label"])

# Scale the features
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df, label_column, test_size=0.3, random_state=42)

# Convert y_train and y_test to 1D arrays
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Apply LDA
lda = LDA(n_components=1)  # Adjust n_components as needed
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Create DataFrame for the reduced features
train_lda_df = pd.DataFrame(X_train_lda, columns=[f'LDA_feature_{i+1}' for i in range(X_train_lda.shape[1])])
train_lda_df['label'] = y_train

test_lda_df = pd.DataFrame(X_test_lda, columns=[f'LDA_feature_{i+1}' for i in range(X_test_lda.shape[1])])
test_lda_df['label'] = y_test

# Save the DataFrames to CSV files
train_lda_df.to_csv('train_lda.csv', index=False)
test_lda_df.to_csv('test_lda.csv', index=False)

print('CSV files saved: train_lda.csv and test_lda.csv')
