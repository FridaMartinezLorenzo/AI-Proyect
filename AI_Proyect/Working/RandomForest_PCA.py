import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

# Load the dataset
EHMS = pd.read_csv('dataset_PCA.csv')
df = pd.DataFrame(EHMS)

# Define features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize KFold cross-validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(rf, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {np.mean(scores)}')

# Train the Random Forest model on the entire dataset
rf.fit(X, y)


print("\nRandom Forest model trained and saved.")
