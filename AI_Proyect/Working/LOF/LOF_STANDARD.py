import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_Standard.csv')
test_data = pd.read_csv('../TrainTest/Split/test_Standard.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label'].values
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label'].values

# Initialize the Local Outlier Factor model with novelty detection enabled
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.15, novelty=True)

# Fit the model on the training data using values
lof.fit(X_train.values, y_train)

# Predict on the test data using values
y_pred_test = lof.predict(X_test.values)

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

# Evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred_test_bin)
print("Accuracy on the test set using Local Outlier Factor:")
print(accuracy)

print("\nClassification report for Local Outlier Factor on test set:")
print(classification_report(y_test, y_pred_test_bin))

# Visualization of the LOF results
lof_labels = np.where(y_pred_test == -1, 'Outlier', 'Inlier')
lof_colors = np.where(y_pred_test == -1, 'red', 'blue')

# Scatter plot to visualize the distribution
plt.figure(figsize=(10, 8))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=lof_colors, label=lof_labels)
plt.title("LOF Distribution of the Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(['Inliers', 'Outliers'])
plt.grid(True)
plt.show()

# Plotly 3D scatter plot for better visualization (if your data has more than two dimensions)
fig = px.scatter_3d(
    x=X_test.iloc[:, 0], 
    y=X_test.iloc[:, 1], 
    z=X_test.iloc[:, 2], 
    color=lof_labels,
    title="3D Scatter Plot of Test Data with LOF",
    labels={'color': 'LOF Label'}
)
fig.show()
