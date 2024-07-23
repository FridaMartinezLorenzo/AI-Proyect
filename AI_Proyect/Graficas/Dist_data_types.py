import pandas as pd
import matplotlib.pyplot as plt


EHMS = pd.read_csv('../WUSTL-EHMS/wustl-ehms-2020.csv')


df = pd.DataFrame(EHMS)


# We manipulated this attribute ‘cause the system detected it as numeric when is categorical
df['Dport'] = df['Dport'].astype('object')


# Identify the numeric and cathegorical attributes
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns


# Count the numeric and cathegorical attributes 
categorical_count = len(categorical_columns)
numerical_count = len(numerical_columns)


# Graph
plt.figure(figsize=(8, 6))
plt.bar(['Categorical', 'Numerical'], [categorical_count, numerical_count], color=['Orange', 'Pink'])
plt.title('Distribution by Category')
plt.xlabel('Feature Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


categorical_columns, numerical_columns
