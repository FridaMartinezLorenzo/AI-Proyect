from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
EHMS = pd.read_csv('../../dataset_pre_processed_balanced_minmax.csv')
df = pd.DataFrame(EHMS)

# Split the dataset into features and labels
X = df.drop(columns=['Label'])
y = df[['Label']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Add the label column to the train and test datasets
X_train = pd.concat([X_train, y_train], axis=1)
X_test = pd.concat([X_test, y_test], axis=1)

#We save on a csv file the train and test datasets
X_train = X_train.to_csv('train_MinMax.csv', index=False)
X_test = X_test.to_csv('test_MinMax.csv', index=False)


print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)
