from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

# Load the dataset
EHMS = pd.read_csv('../../dataset_pre_processed_standard.csv')
df = pd.DataFrame(EHMS)

# Split the dataset into features and labels
X = df
y = df[['Label']]

# Configurar StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Directory to store the fold data
output_dir = "folds_data_Standard"
os.makedirs(output_dir, exist_ok=True)

#Iterate over each fold and save the sets to CSV files
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #Add the label column to the train and test datasets
    X_train = pd.concat([X_train, y_train], axis=1)
    X_test = pd.concat([X_test, y_test], axis=1)

    # Save the training and test sets to CSV files
    X_train.to_csv(os.path.join(output_dir, f"train_fold_{fold + 1}.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, f"test_fold_{fold + 1}.csv"), index=False)
    
    print(f"Fold {fold + 1} saved.")

print("All folds have been saved.")
