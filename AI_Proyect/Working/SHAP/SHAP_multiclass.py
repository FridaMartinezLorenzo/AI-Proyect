import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Cargar el dataset completo
EHMS = pd.read_csv('../dataset_pre_processed_minmax.csv')
df = pd.DataFrame(EHMS)

X = df.drop(columns=['Label'])
y = df['Label']

# Cargar los datasets de entrenamiento y prueba
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

# Dividir los datos en características y etiquetas
X_train = train_data.drop(columns=['Label'])
X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
y_test = test_data['Label']

# Convertir y a un array unidimensional
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Entrenar el modelo Random Forest
rf = RandomForestClassifier(n_estimators=166,
                            max_depth=10,
                            min_samples_split=9,
                            min_samples_leaf=3,
                            max_features='log2',
                            bootstrap=True,
                            random_state=42)
rf.fit(X_train, y_train)

# Imprimir la precisión del entrenamiento
print(f'Training accuracy: {rf.score(X_train, y_train)}')

# Importancia de las características
importances = rf.feature_importances_
index =  np.argsort(importances)
features = df.columns

plt.title('Feature Importances')
plt.barh(range(len(index)), importances[index], color='g', align='center')
plt.yticks(range(len(index)), [features[i] for i in index], fontsize=5)
plt.xlabel('Relative Importance')
plt.show()

# Matriz de confusión
class_names = np.unique(y)
disp = ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

# Imprimir la distribución de las etiquetas
print(y.value_counts())

# SHAP summary plot

# Calcular los valores SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X)

print("Shap dimension:", shap_values.shape)  # Debería ser (16318, 41, 2) si hay dos clases
print("X dimension:", X.shape)  # Debería ser (16318, 41)

# Visualizar el SHAP waterfall plot para la primera instancia de la clase 0 (no ataque)
#shap.plots.waterfall(shap_values[0][0])

# Si quieres visualizar la explicación SHAP para la clase 1 (ataque), usa:
# shap.plots.waterfall(shap_values[1][0])

#waterfall plot for class 0
shap.plots.waterfall(shap_values[0,:,0])

# waterfall plot for class 1
shap.plots.waterfall(shap_values[0,:,1])
