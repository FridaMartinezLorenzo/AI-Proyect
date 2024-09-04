import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Extract features and labels from the train and test datasets
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label'].values
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label'].values

# Initialize the Local Outlier Factor model with novelty detection enabled
lof = LocalOutlierFactor(n_neighbors=35, contamination=0.15, novelty=True)

# Fit the model on the training data
lof.fit(X_train.values)

# Predict on the test data
y_pred_test = lof.predict(X_test.values)
X_scores = -lof.decision_function(X_test.values)  # LOF outlier scores

# Convert predictions (-1 for outliers, 1 for inliers) to binary format
y_pred_test_bin = np.where(y_pred_test == -1, 0, 1)

# Evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred_test_bin)
print("Accuracy on the test set using Local Outlier Factor:")
print(accuracy)

print("\nClassification report for Local Outlier Factor on test set:")
print(classification_report(y_test, y_pred_test_bin))

# Custom function to update legend marker size
def update_legend_marker_size(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([20])

# Scatter plot to visualize LOF with outlier scores
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], color="k", s=3.0, label="Data points")

# Plot circles with radii proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X_test.iloc[:, 0],
    X_test.iloc[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

plt.axis("tight")
plt.xlim((X_test.iloc[:, 0].min(), X_test.iloc[:, 0].max()))
plt.ylim((X_test.iloc[:, 1].min(), X_test.iloc[:, 1].max()))
plt.xlabel(f"Prediction errors: {sum(y_pred_test_bin != y_test)}")
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF) Visualization")
plt.show()
