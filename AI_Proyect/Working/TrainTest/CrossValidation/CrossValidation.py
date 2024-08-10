from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Supongamos que tienes un DataFrame de pandas llamado 'data'
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Dividir los datos en características (X) y etiquetas (y)
X = data[['feature1', 'feature2']]
y = data['label']

# Configurar StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterar sobre cada fold
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f"Fold {fold + 1}")
    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:\n", y_train)
    print("y_test:\n", y_test)
    print("-" * 30)
