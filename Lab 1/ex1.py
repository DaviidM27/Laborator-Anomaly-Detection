import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

contamination = 0.1

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=contamination
)

normal = y_train == 0
outliers = y_train == 1

plt.figure(figsize=(8, 6))

plt.scatter(
    X_train[normal, 0],
    X_train[normal, 1],
    c="blue",
    label="Normal samples",
    alpha=0.7
)

plt.scatter(
    X_train[outliers, 0],
    X_train[outliers, 1],
    c="red",
    label="Outliers",
    marker="x",
    s=80
)

plt.title("Dataset Representation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("Normal training samples:", np.sum(y_train == 0))
print("Training outliers:", np.sum(y_train == 1))

print("Normal test samples:", np.sum(y_test == 0))
print("Test outliers:", np.sum(y_test == 1))