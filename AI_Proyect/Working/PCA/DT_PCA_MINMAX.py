import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Cargar los conjuntos de datos de entrenamiento y prueba
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Extraer características y etiquetas de los conjuntos de datos
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Imputar valores faltantes en las características con la media
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Manejar valores faltantes en las etiquetas
# Opción 1: Eliminar filas con etiquetas faltantes
non_nan_train_indices = ~y_train.isna()
non_nan_test_indices = ~y_test.isna()

X_train = X_train[non_nan_train_indices]
y_train = y_train[non_nan_train_indices]

X_test = X_test[non_nan_test_indices]
y_test = y_test[non_nan_test_indices]

# Aplicar PCA a los datos de entrenamiento para determinar el número de componentes
pca = PCA()
pca.fit(X_train)

# Seleccionar el número de componentes principales para explicar el 95% de la varianza
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.95) + 1

print(f'Número de componentes principales para explicar el 95% de la varianza: {n_components}')

# Inicializar PCA con el número seleccionado de componentes
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Matriz de componentes principales (cargas)
loading_scores = pca.components_

loading_scores_df = pd.DataFrame(loading_scores, columns=train_data.columns[:-1])

# Mostrar las características que más contribuyen a cada componente principal
for i in range(n_components):
    pc_contribution = loading_scores_df.iloc[i].abs().sort_values(ascending=False)
    print(f"Principales características que contribuyen a PC{i+1}:")
    print(pc_contribution.head(10))
    print("\n")

# Inicializar el clasificador de árbol de decisión
dt = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo de árbol de decisión en el conjunto de entrenamiento
dt.fit(X_train_pca, y_train)

# Predecir en el conjunto de prueba
y_pred = dt.predict(X_test_pca)
y_prob = dt.predict_proba(X_test_pca)[:, 1]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Imprimir métricas de evaluación
print(f'Precisión en el conjunto de prueba: {accuracy}')
print(f'F1-score en el conjunto de prueba: {f1}')
print(f'AUC score en el conjunto de prueba: {auc}')
print(f'Precisión en el conjunto de prueba: {precision}')
print(f'Recall en el conjunto de prueba: {recall}')
