import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Load the training and testing datasets
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Split the data into features and labels
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Convert y to a one-dimensional array
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Get the feature names
feature_names = X_train.columns

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=166,
                            max_depth=10,
                            min_samples_split=9,
                            min_samples_leaf=3,
                            max_features='log2',
                            bootstrap=True,
                            random_state=42)
rf.fit(X_train, y_train)

# Get the feature importances
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Select the most important features
threshold = 0.01  # Importance threshold
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filter the dataset with the valid features using Pandas DataFrame selection
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Calculate the correlation matrix for important features
correlation_matrix = X_train_important.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix of Important Features", fontsize=12)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.show()


#_______________________________________________________________________________
# Train the Random Forest model again with the important features
rf_important = RandomForestClassifier(n_estimators=166,
                                      max_depth=10,
                                      min_samples_split=9,
                                      min_samples_leaf=3,
                                      max_features='log2',
                                      bootstrap=True,
                                      random_state=42)
rf_important.fit(X_train_important, y_train)

#_______________________________________________________________________________
# Get the unique class names from y_test
class_names = np.unique(y_test)

print("\nY Test:", y_test)
print("\n\nClass names:", class_names)


# Plot the confusion matrix using the important features
disp = ConfusionMatrixDisplay.from_estimator(
    estimator=rf_important,
    X=X_test_important,
    y=y_test,
    display_labels=class_names,  # Utilizar las clases únicas de y_test
    cmap=plt.cm.Blues,
    xticks_rotation='vertical'
)
plt.title('Confusion Matrix of Important Features')
plt.show()

"""_______________________________________________________________________________
SHAP Summary Plot
SHAP values of a model’s output explain how features impact the output of the model.
_______________________________________________________________________________"""
# Compute SHAP values
explainer = shap.TreeExplainer(rf_important)
shap_values = explainer.shap_values(X_test_important)

print("SHAP values:", shap_values)
print("X_test_important:", X_test_important)

shap.summary_plot(shap_values, X_test_important, plot_type="bar", class_names= class_names, feature_names = X_test_important.columns)
shap.summary_plot(shap_values[0], X_test_important.values, plot_type="bar", class_names=class_names, feature_names=X_test_important.columns)

# Plot SHAP summary plot
#shap.summary_plot(shap_values, X_test_important, feature_names=important_features, class_names=np.unique(y_test))
