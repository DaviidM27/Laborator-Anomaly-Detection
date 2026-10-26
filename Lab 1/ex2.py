import numpy as np
import matplotlib.pyplot as plt

from pyod.utils.data import generate_data
from pyod.models.knn import KNN

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score
)

contamination = 0.1

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=contamination
)

model = KNN(contamination=contamination)
model.fit(X_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)

print("\nTrue Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives (TP):", tp)

ba = balanced_accuracy_score(y_test, y_test_pred)

print("\nBalanced Accuracy:", round(ba, 4))

y_test_scores = model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
auc = roc_auc_score(y_test, y_test_scores)

print("AUC:", round(auc, 4))

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

ba_train = balanced_accuracy_score(y_train, y_train_pred)

print("\nTraining Balanced Accuracy:", round(ba_train, 4))
print("Test Balanced Accuracy:", round(ba, 4))

print("\nEffect of Different Contamination Rates:")

for contamination_value in [0.05, 0.1, 0.2, 0.3]:

    model = KNN(contamination=contamination_value)
    model.fit(X_train)

    y_pred = model.predict(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)

    n_predicted_outliers = np.sum(y_pred == 1)

    print(
        f"Contamination = {contamination_value:.2f} | "
        f"Balanced Accuracy = {ba:.4f} | "
        f"Predicted Outliers = {n_predicted_outliers}"
    )