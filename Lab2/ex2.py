from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400, n_test=200, n_clusters=2, n_features=2, 
    contamination=0.1, size='same', density='same', dist=0.25
)

n_neighbors_list = [5, 10, 20]
for n_neighbors in n_neighbors_list:
    clf = KNN(n_neighbors=n_neighbors)
    clf.fit(X_train)
    
    y_train_pred = clf.labels_
    y_test_pred = clf.predict(X_test)
    
    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors={n_neighbors} | BA Train: {ba_train:.3f} | BA Test: {ba_test:.3f}")
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
    axs[0, 0].set_title("Ground Truth - Training Data")
    
    axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm')
    axs[0, 1].set_title("Predicted - Training Data")
    
    axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
    axs[1, 0].set_title("Ground Truth - Test Data")
    
    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm')
    axs[1, 1].set_title("Predicted - Test Data")
    
    plt.tight_layout()
    plt.show()