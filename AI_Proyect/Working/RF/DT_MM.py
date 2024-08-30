import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

# Cargar los conjuntos de entrenamiento y prueba
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Dividir los datos en características y etiquetas
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Manejar valores faltantes en las características
imputer_X = SimpleImputer(strategy='mean')
X_train_imputed = imputer_X.fit_transform(X_train)
X_test_imputed = imputer_X.transform(X_test)

# Manejar valores faltantes en las etiquetas
non_nan_train_indices = ~y_train.isna()
non_nan_test_indices = ~y_test.isna()

X_train_imputed = X_train_imputed[non_nan_train_indices]
y_train = y_train[non_nan_train_indices]

X_test_imputed = X_test_imputed[non_nan_test_indices]
y_test = y_test[non_nan_test_indices]

# Convertir y a un arreglo unidimensional
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Obtener los nombres de las características
feature_names = X_train.columns

# Entrenar el modelo de Random Forest
rf = RandomForestClassifier(n_estimators=166,
                            max_depth=10,
                            min_samples_split=9,
                            min_samples_leaf=3,
                            max_features='log2',
                            bootstrap=True,
                            random_state=42)
rf.fit(X_train_imputed, y_train)

# Obtener la importancia de las características
feature_importances = rf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances_df)

# Guardar la importancia de las características en un archivo CSV
try:
    feature_importances_df.to_csv('feature_importances1_df.csv', index=False)
    print("Feature importances saved successfully to 'feature_importances1_df.csv'")
except Exception as e:
    print(f"Error saving feature importances: {e}")

# Seleccionar las características más importantes
threshold = 0.01  # Umbral de importancia
important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]['Feature']
print("Most important features:", important_features.tolist())

# Filtrar solo las características que existen en los datos de entrenamiento
valid_important_features = [feature for feature in important_features if feature in feature_names]
print("Valid features:", valid_important_features)

# Filtrar el conjunto de datos con las características válidas
X_train_important = X_train_imputed[:, [list(feature_names).index(feature) for feature in valid_important_features]]
X_test_important = X_test_imputed[:, [list(feature_names).index(feature) for feature in valid_important_features]]

# Contar el número de características biométricas
biometric_columns = ['Temp', 'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST']
number_of_biometric_features = len(np.intersect1d(valid_important_features, biometric_columns))

print("\n______________________________________________________")
print("Counting the number of biometric features")
print("______________________________________________________")
print("\nNumber of biometric features:", number_of_biometric_features)
print("Number of network flow features:", len(valid_important_features) - number_of_biometric_features)

# Guardar el conjunto de datos filtrado en un nuevo archivo CSV
X_train_important_df = pd.DataFrame(X_train_important, columns=valid_important_features)
X_test_important_df = pd.DataFrame(X_test_important, columns=valid_important_features)

try:
    X_train_important_df.to_csv('train_filtered_dataset.csv', index=False)
    X_test_important_df.to_csv('test_filtered_dataset.csv', index=False)
    print("\nFiltered datasets saved successfully to 'train_filtered_dataset.csv' and 'test_filtered_dataset.csv'")
except Exception as e:
    print(f"\nError saving filtered datasets: {e}")

# Entrenar un modelo de Decision Tree con las características seleccionadas
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_important, y_train)

# Predecir y evaluar el modelo
y_pred = dt_classifier.predict(X_test_important)
y_pred_proba = dt_classifier.predict_proba(X_test_important)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Calcular el AUC para cada clase y luego promediar
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
if y_test_binarized.shape[1] == y_pred_proba.shape[1]:
    auc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
else:
    auc = None  # AUC no es aplicable para escenarios binarios o de una sola clase

print(f"Model accuracy with selected features: {accuracy}")
print(f"Model F1-score with selected features: {f1}")
print(f"Model precision with selected features: {precision}")
print(f"Model recall with selected features: {recall}")
if auc is not None:
    print(f"Model AUC with selected features: {auc}")
else:
    print("AUC is not applicable for this dataset.")
