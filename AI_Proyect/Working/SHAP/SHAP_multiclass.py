import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

#Load the entire dataset
EHMS = pd.read_csv('../dataset_pre_processed_minmax.csv')
df = pd.DataFrame(EHMS)

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

print(f'Training accuracy: {rf.score(X_train, y_train)}')

# Get the feature importances
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Select the most important features
threshold = 0.01  # Importance threshold
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filter the dataset with the valid features using Pandas DataFrame selection
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Train the Random Forest model again with the important features
rf_important = RandomForestClassifier(n_estimators=166,
                                      max_depth=10,
                                      min_samples_split=9,
                                      min_samples_leaf=3,
                                      max_features='log2',
                                      bootstrap=True,
                                      random_state=42)
rf_important.fit(X_train_important, y_train)

# Class names for confusion matrix
class_names = np.unique(y_test)


# Display the value counts for the target variable
#print(y_test.value_counts())

# Compute SHAP values
explainer = shap.TreeExplainer(rf_important)
shap_values = explainer.shap_values(X_train)

# Generate SHAP summary plot
# If you want to plot for each class separately, you can do that as well:
for i in range(len(class_names)):
    shap.summary_plot(shap_values[i], X_test, plot_type="bar", class_names=class_names, feature_names=X_test.columns, show=False)
    plt.title(f'SHAP Summary for Class: {class_names[i]}')
    plt.show()
    

shap.summary_plot(shap_values[1], X_test.values, feature_names = X_test.columns)

"""
print(shap_values)

# Check the correct shape of SHAP values
for i, class_name in enumerate(class_names):
    print(f"Plotting SHAP summary for class: {class_name}")
    
    print(f"SHAP values shape for class {class_name}: {shap_values[i].shape}")
    print(f"X_test_important shape: {X_test_important.shape}") 
    
    # Plotting SHAP summary if shapes match
    if shap_values[i].shape == X_test_important.shape:
        shap.summary_plot(shap_values[i], X_test_important, plot_type="bar", feature_names=X_test_important.columns)
    else:
        print(f"Shape mismatch for class {class_name}: Cannot plot SHAP summary.")

"""