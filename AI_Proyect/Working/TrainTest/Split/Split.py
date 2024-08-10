from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd


# Load the dataset
EHMS = pd.read_csv('../../../WUSTL-EHMS/wustl-ehms-2020.csv')
df = pd.DataFrame(EHMS)

# Drop unnecessary columns
df = df.drop(['Dir', 'Flgs'], axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Convert 'Dport' to an object type
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

# Aply the Min-Max Scaling to all the columns
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split the dataset into training and testing sets
X = df
y = label_column


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Add the label column to the train and test datasets
X_train = pd.concat([X_train, label_column], axis=1)
X_test = pd.concat([X_test, label_column], axis=1)

#We save on a csv file the train and test datasets
X_train = X_train.to_csv('train_MinMax.csv', index=False)
X_test = X_test.to_csv('test_MinMax.csv', index=False)


print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)
