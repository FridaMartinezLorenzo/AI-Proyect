import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

label_column = df['Label']
df = df.drop(['Label'], axis=1)

# Check categorical columns
print("Categorical columns before One-Hot Encoding:", categorical_columns)

# Apply One-Hot Encoding on categorical columns
df = pd.get_dummies(df, columns=categorical_columns)

# Convert boolean columns (0, 1) to int type
df = df.astype(int)

# Verify the conversion
print("Columns after One-Hot Encoding:", df.columns)

# Verify the number of columns
print("Number of columns after One-Hot Encoding:", len(df.columns))

# Select only numeric columns to calculate correlation
df_numeric = df.select_dtypes(include=[np.number])

print("Number of numeric columns after One-Hot Encoding:", len(df_numeric.columns))

# Set the general font size
sns.set(font_scale=0.7)

# Correlation matrix before One-Hot Encoding
df_original_numeric = df_original.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix before One-Hot Encoding", fontsize=10)
sns.heatmap(df_original_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_no_OHE.png')
plt.show()

# Correlation matrix after One-Hot Encoding
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix after One-Hot Encoding", fontsize=10)
sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 4}, cmap='coolwarm')
plt.savefig('Dist_OHE.png')
plt.show()
