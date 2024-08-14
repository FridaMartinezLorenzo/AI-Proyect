import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# Load the training and test datasets
# Cargar los datasets de entrenamiento y prueba
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Dividir los datos en características y etiquetas
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']


# Definir el modelo
rf = RandomForestClassifier(random_state=42)

# Definir la rejilla de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],          # Número de árboles en el bosque
    'max_depth': [None, 10, 20, 30],         # Máxima profundidad de los árboles
    'min_samples_split': [2, 5, 10],         # Número mínimo de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4],           # Número mínimo de muestras requeridas en un nodo hoja
    'max_features': ['auto', 'sqrt', 'log2'],# Número de características a considerar al buscar la mejor división
    'bootstrap': [True, False]               # Si se utiliza o no bootstrap al construir árboles
}


# Configurar la búsqueda con validación cruzada
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Entrenar el modelo con la mejor combinación de hiperparámetros
grid_search.fit(X_train, y_train)

# Obtener la mejor combinación de hiperparámetros
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)


# Evaluar el modelo con los datos de prueba usando el mejor modelo
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# Calcular las métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Calcular AUC para cada clase y luego promediar
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
if y_test_binarized.shape[1] == y_pred_proba.shape[1]:
    auc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
else:
    auc = None  # AUC no es aplicable para escenarios binarios o de una sola clase

print(f"Model accuracy with optimized hyperparameters: {accuracy}")
print(f"Model F1-score with optimized hyperparameters: {f1}")
print(f"Model precision with optimized hyperparameters: {precision}")
print(f"Model recall with optimized hyperparameters: {recall}")
if auc is not None:
    print(f"Model AUC with optimized hyperparameters: {auc}")
else:
    print("AUC is not applicable for this dataset.")
