import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Cargar los datasets de entrenamiento y prueba
#train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
#test_data = pd.read_csv('../TrainTest/Split/test_MinMax.csv')

#Trabajamos con el dataset preprocesado
train_data = pd.read_csv('../dataset_pre_processed_minmax.csv')

# Separar las características y las etiquetas
X_train = train_data.drop(columns=['Label'])
#X_test = test_data.drop(columns=['Label'])
y_train = train_data['Label']
#y_test = test_data['Label']


# Aplicar t-SNE para reducir las dimensiones a 2 componentes
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_train_tsne = tsne.fit_transform(X_train)

# Crear un DataFrame para los datos transformados
tsne_df = pd.DataFrame(X_train_tsne, columns=['Component 1', 'Component 2'])
tsne_df['Label'] = y_train.values

# Visualizar los resultados de t-SNE
plt.figure(figsize=(10, 8))
for label in np.unique(y_train):
    subset = tsne_df[tsne_df['Label'] == label]
    plt.scatter(subset['Component 1'], subset['Component 2'], label=f'Class {label}', s=50)

plt.title('t-SNE Visualization of the Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.grid(True)
plt.show()
