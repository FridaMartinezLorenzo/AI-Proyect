import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Trabajamos con el dataset preprocesado
train_data = pd.read_csv('../dataset_pre_processed_standard.csv')

# Separar las características y las etiquetas
X = train_data.drop(columns=['Label'])
y = train_data['Label']

# Aplicar t-SNE para reducir las dimensiones a 2 componentes
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# Crear un DataFrame para los datos transformados
tsne_df = pd.DataFrame(X_tsne, columns=['Component 1', 'Component 2'])
tsne_df['Label'] = y.values

# Visualizar los resultados de t-SNE con Matplotlib
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    subset = tsne_df[tsne_df['Label'] == label]
    plt.scatter(subset['Component 1'], subset['Component 2'], label=f'Class {label}', s=50)

plt.title('t-SNE Visualization of the Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Visualización 3D con Plotly (original data)
fig = px.scatter_3d(x=X.iloc[:, 0], y=X.iloc[:, 1], z=X.iloc[:, 2], color=y, opacity=0.8)
fig.update_layout(title='3D Scatter Plot of Original Data')
fig.show()

# Visualización 2D del conjunto después de aplicar t-SNE con Plotly
fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y)
fig.update_layout(
    title="t-SNE Visualization of Custom Classification Dataset",
    xaxis_title="First t-SNE Component",
    yaxis_title="Second t-SNE Component",
)
fig.show()

# Entrenamiento del modelo de Random Forest usando los resultados de t-SNE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tsne, y)

# Predicción y evaluación
y_pred = rf.predict(X_tsne)
y_prob = rf.predict_proba(X_tsne)[:, 1]

# Evaluación del modelo
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# Mostrar las métricas de evaluación
print(f'Accuracy on the t-SNE transformed data: {accuracy}')
print(f'F1-score on the t-SNE transformed data: {f1}')
print(f'AUC score on the t-SNE transformed data: {auc}')
print(f'Precision on the t-SNE transformed data: {precision}')
print(f'Recall on the t-SNE transformed data: {recall}')
