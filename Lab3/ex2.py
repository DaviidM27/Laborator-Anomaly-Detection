import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch 
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_2d, _ = make_blobs(n_samples=1000, centers=[[10, 0], [0, 10]], cluster_std=1.0)
X_test_2d = np.random.uniform(low=-10, high=20, size=(1000, 2))

models_2d = {
    'IForest': IForest(contamination=0.02),
    'DIF': DIF(contamination=0.02, hidden_neurons=[16, 16]),
    'LODA': LODA(contamination=0.02, n_bins=20)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, clf) in zip(axes, models_2d.items()):
    clf.fit(X_train_2d)
    scores_2d = clf.decision_function(X_test_2d)
    sc = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=scores_2d, cmap='viridis')
    ax.set_title(f"{name} (2D)")
    fig.colorbar(sc, ax=ax)
plt.suptitle('Comparing anomaly scores (2D)')
plt.show()

X_train_3d, _ = make_blobs(n_samples=1000, centers=[[0, 10, 0], [10, 0, 10]], cluster_std=1.0)
X_test_3d = np.random.uniform(low=-10, high=20, size=(1000, 3))

models_3d = {
    'IForest': IForest(contamination=0.02),
    'DIF': DIF(contamination=0.02, hidden_neurons=[16, 16], device=device),
    'LODA': LODA(contamination=0.02, n_bins=20)
}

fig_3d = plt.figure(figsize=(18, 5))
for i, (name, clf) in enumerate(models_3d.items(), 1):
    clf.fit(X_train_3d)
    scores_3d = clf.decision_function(X_test_3d)
    
    ax = fig_3d.add_subplot(1, 3, i, projection='3d')
    sc = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=scores_3d, cmap='viridis')
    ax.set_title(f"{name} (3D)")
    fig_3d.colorbar(sc, ax=ax)
plt.suptitle('Comparing anomaly scores (3D)')
plt.show()