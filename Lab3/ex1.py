import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X_train, _ = make_blobs(n_samples=500, n_features=2, centers=[[0, 0]], cluster_std=1.0)

n_projections = 5
W = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_projections)
W = W / np.linalg.norm(W, axis=1)[:, np.newaxis]

n_bins = 20
histograms = []
bin_edges_list = []

for i in range(n_projections):
    proj_train = np.dot(X_train, W[i])
    min_val, max_val = proj_train.min() - 2, proj_train.max() + 2
    
    counts, bin_edges = np.histogram(proj_train, bins=n_bins, range=(min_val, max_val))
    probs = counts / counts.sum()
    
    histograms.append(probs)
    bin_edges_list.append(bin_edges)

X_test = np.random.uniform(low=-3, high=3, size=(500, 2))
mean_probabilities = np.zeros(500)

for i in range(n_projections):
    proj_test = np.dot(X_test, W[i])
    bin_indices = np.digitize(proj_test, bin_edges_list[i]) - 1
    
    prob_test = np.zeros(len(proj_test))
    for j in range(len(proj_test)):
        if 0 <= bin_indices[j] < n_bins:
            prob_test[j] = histograms[i][bin_indices[j]]
            
    mean_probabilities += prob_test

mean_probabilities /= n_projections
anomaly_scores = 1 - mean_probabilities 

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_test[:, 0], X_test[:, 1], c=anomaly_scores, cmap='coolwarm')
plt.colorbar(sc, label='Anomaly Score (1 - Probability)')
plt.title(f'LODA Simplified (Bins={n_bins})')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()