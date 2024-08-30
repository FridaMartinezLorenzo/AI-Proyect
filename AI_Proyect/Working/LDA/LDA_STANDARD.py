#LDA WITH STANDARD SCALING

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the dataset
EHMS = pd.read_csv('../dataset_pre_processed_standard.csv')
df = pd.DataFrame(EHMS)

# Split the dataset into training and test sets
X_train = pd.read_csv('../TrainTest/Split/train_Standard.csv')
X_test = pd.read_csv('../TrainTest/Split/test_Standard.csv')
y_train = X_train[['Label']]
y_test = X_test[['Label']]
X_train = X_train.drop(columns=['Label'])
X_test = X_test.drop(columns=['Label'])

# Convert y_train and y_test to 1D arrays
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Apply LDA
lda = LDA(n_components=1)  # Adjust n_components as needed
lda.fit(X_train, y_train)

# Get the feature coefficients
feature_importance = lda.coef_[0]

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': feature_importance
}).sort_values(by='Coefficient', ascending=False)

# Display the features and their corresponding coefficients
#print(importance_df)

# Display the top 10 features and their corresponding coefficients
top_10_features = importance_df.head(10)
print(top_10_features)


# Transform the dataset using LDA
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Create DataFrame for the reduced features
train_lda_df = pd.DataFrame(X_train_lda, columns=[f'LDA_feature_{i+1}' for i in range(X_train_lda.shape[1])])
train_lda_df['Label'] = y_train

test_lda_df = pd.DataFrame(X_test_lda, columns=[f'LDA_feature_{i+1}' for i in range(X_test_lda.shape[1])])
test_lda_df['Label'] = y_test
# Save the DataFrames to CSV files
train_lda_df.to_csv('train_lda1.csv', index=False)
test_lda_df.to_csv('test_lda1.csv', index=False)

