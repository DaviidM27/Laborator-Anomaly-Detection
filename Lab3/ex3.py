import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

n_splits = 10
results = {
    'IForest': {'ROC_AUC': [], 'BA': []},
    'LODA': {'ROC_AUC': [], 'BA': []},
    'DIF': {'ROC_AUC': [], 'BA': []}
}

for i in range(n_splits):
    print(f"Split {i + 1}/{n_splits}...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train_norm, X_test_norm = standardizer(X_train, X_test)
    
    models = {
        'IForest': IForest(),
        'LODA': LODA(),
        'DIF': DIF(device=device) 
    }
    
    for name, clf in models.items():
        clf.fit(X_train_norm)
        
        y_test_scores = clf.decision_function(X_test_norm)
        y_test_pred = clf.predict(X_test_norm)
        
        roc_auc = roc_auc_score(y_test, y_test_scores)
        ba = balanced_accuracy_score(y_test, y_test_pred)
        
        results[name]['ROC_AUC'].append(roc_auc)
        results[name]['BA'].append(ba)

print("-" * 50)
for name in results:
    mean_roc = np.mean(results[name]['ROC_AUC'])
    mean_ba = np.mean(results[name]['BA'])
    print(f"Model: {name}")
    print(f"Mean ROC AUC: {mean_roc:.4f}")
    print(f"Mean Balanced Accuracy: {mean_ba:.4f}")
    print("-" * 50)