import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

np.random.seed(42)

n_samples = 1000
contamination = 0.1

n_outliers = int(n_samples * contamination)
n_normal = n_samples - n_outliers

mu = np.array([2.0, 3.0])

Sigma = np.array([
    [1.0, 0.6],
    [0.6, 2.0]
])

L = np.linalg.cholesky(Sigma)

print("Mean Vector (mu):")
print(mu)

print("\nCovariance Matrix (Sigma):")
print(Sigma)

print("\nCholesky Matrix (L):")
print(L)

print("\nVerification of L @ L.T:")
print(L @ L.T)

X_standard = np.random.randn(n_normal, 2)

X_normal = X_standard @ L.T + mu

X_outliers_standard = np.random.randn(n_outliers, 2)

X_outliers = (
    X_outliers_standard @ L.T
    + mu
    + np.array([6.0, 6.0])
)

X = np.vstack([X_normal, X_outliers])

y_true = np.concatenate([
    np.zeros(n_normal, dtype=int),
    np.ones(n_outliers, dtype=int)
])

indices = np.random.permutation(n_samples)

X = X[indices]
y_true = y_true[indices]

feature_std = np.sqrt(np.diag(Sigma))

z_scores = (X - mu) / feature_std

anomaly_scores = np.max(np.abs(z_scores), axis=1)

threshold = np.quantile(
    anomaly_scores,
    1 - contamination
)

y_pred = (anomaly_scores >= threshold).astype(int)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

ba = balanced_accuracy_score(y_true, y_pred)

print("\nAnomaly Threshold:", round(threshold, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)

print("\nBalanced Accuracy:", round(ba, 4))

plt.figure(figsize=(9, 7))

normal_pred = y_pred == 0
outliers_pred = y_pred == 1

plt.scatter(
    X[normal_pred, 0],
    X[normal_pred, 1],
    c="blue",
    label="Predicted Normal Samples",
    alpha=0.7
)

plt.scatter(
    X[outliers_pred, 0],
    X[outliers_pred, 1],
    c="red",
    label="Predicted Anomalies",
    marker="x",
    s=60
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multivariate Anomaly Detection")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 7))

normal_true = y_true == 0
outliers_true = y_true == 1

plt.scatter(
    X[normal_true, 0],
    X[normal_true, 1],
    c="blue",
    label="True Normal Samples",
    alpha=0.7
)

plt.scatter(
    X[outliers_true, 0],
    X[outliers_true, 1],
    c="red",
    label="True Anomalies",
    marker="x",
    s=60
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Ground-Truth Labels")
plt.legend()
plt.grid(True)
plt.show()