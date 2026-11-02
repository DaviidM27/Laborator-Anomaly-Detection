import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization

mat = scipy.io.loadmat('cardio.mat')
X = mat['X']
y = mat['y'].ravel()

contamination = np.count_nonzero(y) / len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

n_models = 10
train_scores = np.zeros((X_train_norm.shape[0], n_models))
test_scores = np.zeros((X_test_norm.shape[0], n_models))

for i, n_neighbors in enumerate(range(30, 130, 10)):
    clf = KNN(n_neighbors=n_neighbors)
    clf.fit(X_train_norm)
    
    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test_norm)
    
    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors={n_neighbors} | BA Train: {ba_train:.3f} | BA Test: {ba_test:.3f}")
    
    train_scores[:, i] = clf.decision_scores_
    test_scores[:, i] = clf.decision_function(X_test_norm)

train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)

y_train_avg = average(train_scores_norm)
y_test_avg = average(test_scores_norm)

y_train_max = maximization(train_scores_norm)
y_test_max = maximization(test_scores_norm)

threshold_avg = np.quantile(y_train_avg, 1 - contamination)
threshold_max = np.quantile(y_train_max, 1 - contamination)

y_pred_avg = (y_test_avg >= threshold_avg).astype(int)
y_pred_max = (y_test_max >= threshold_max).astype(int)

ba_avg = balanced_accuracy_score(y_test, y_pred_avg)
ba_max = balanced_accuracy_score(y_test, y_pred_max)

print(f"BA Average Strategy: {ba_avg:.3f}")
print(f"BA Maximization Strategy: {ba_max:.3f}")