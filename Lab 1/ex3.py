import numpy as np
import matplotlib.pyplot as plt

from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

contamination = 0.1
n_samples = 1000

X_train, y_train = generate_data(
    n_train=n_samples,
    n_features=1,
    contamination=contamination,
    train_only=True
)

X = X_train.ravel()

mu = np.mean(X)
sigma = np.std(X)

z_scores = (X - mu) / sigma

z_scores_abs = np.abs(z_scores)

threshold = np.quantile(
    z_scores_abs,
    1 - contamination
)

y_pred = (z_scores_abs >= threshold).astype(int)

cm = confusion_matrix(
    y_train,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

ba = balanced_accuracy_score(y_train, y_pred)

print("Mean:", round(mu, 4))
print("Standard Deviation:", round(sigma, 4))
print("Z-score Threshold:", round(threshold, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)

print("\nBalanced Accuracy:", round(ba, 4))

plt.figure(figsize=(10, 6))

sample_indices = np.arange(len(X))

plt.scatter(
    sample_indices[y_pred == 0],
    X[y_pred == 0],
    c="blue",
    label="Normal Samples",
    alpha=0.7
)

plt.scatter(
    sample_indices[y_pred == 1],
    X[y_pred == 1],
    c="red",
    label="Detected Anomalies",
    marker="x",
    s=60
)

plt.axhline(
    mu + threshold * sigma,
    linestyle="--",
    label="Upper Threshold"
)

plt.axhline(
    mu - threshold * sigma,
    linestyle="--",
    label="Lower Threshold"
)

plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("Anomaly Detection Using Z-Scores")
plt.legend()
plt.grid(True)
plt.show()