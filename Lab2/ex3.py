from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6])

n_neighbors = 20
contamination = 0.07

clf_knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
clf_knn.fit(X)
y_pred_knn = clf_knn.labels_

clf_lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
clf_lof.fit(X)
y_pred_lof = clf_lof.labels_

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(X[:, 0], X[:, 1], c=y_pred_knn, cmap='coolwarm')
axs[0].set_title("KNN Predictions")

axs[1].scatter(X[:, 0], X[:, 1], c=y_pred_lof, cmap='coolwarm')
axs[1].set_title("LOF Predictions")

plt.show()