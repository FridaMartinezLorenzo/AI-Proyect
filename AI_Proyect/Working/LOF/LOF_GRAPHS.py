import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Load your dataset
train_data = pd.read_csv('../TrainTest/Split/train_MinMax.csv')
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']

# Set the outlier fraction
outliers_fraction = 0.15

# Reduce data dimensionality to 2D using PCA for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)

# Define the anomaly detection algorithms
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction, random_state=42)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    ("One-Class SVM (SGD)", make_pipeline(Nystroem(gamma=0.1, random_state=42, n_components=150),
                                          SGDOneClassSVM(nu=outliers_fraction, shuffle=True, 
                                                         fit_intercept=True, random_state=42, tol=1e-6))),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)),
]

# Create a mesh grid for contour plotting
xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))

# Plot settings
plt.figure(figsize=(len(anomaly_algorithms) * 2 + 4, 12.5))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01)

plot_num = 1

for name, algorithm in anomaly_algorithms:
    t0 = time.time()

    if name == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(X_reduced)
    else:
        algorithm.fit(X_reduced)
        y_pred = algorithm.predict(X_reduced)

    t1 = time.time()

    plt.subplot(1, len(anomaly_algorithms), plot_num)
    plt.title(name, size=18)

    # Plot the levels lines and the points
    if name != "Local Outlier Factor":  # LOF does not implement predict
        Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

    colors = np.array(["#377eb8", "#ff7f00"])
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, color=colors[(y_pred + 1) // 2])

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.xticks(())
    plt.yticks(())
    plt.text(0.99, 0.01, ("%.2fs" % (t1 - t0)).lstrip("0"), transform=plt.gca().transAxes,
             size=15, horizontalalignment="right")

    plot_num += 1

plt.show()
