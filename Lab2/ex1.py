import numpy as np
import matplotlib.pyplot as plt

def leverage_scores(X):
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return np.diag(H)

a, b, c_val = 2.0, 1.0, 0.5
n_samples = 200
variances = [0.1, 1.0, 5.0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, var in zip(axes, variances):
    X_reg = np.random.uniform(0, 10, int(n_samples * 0.85))
    y_reg = a * X_reg + b + np.random.normal(0, np.sqrt(var), len(X_reg))
    
    X_hx = np.random.uniform(15, 20, int(n_samples * 0.05))
    y_hx = a * X_hx + b + np.random.normal(0, np.sqrt(var), len(X_hx))
    
    X_hy = np.random.uniform(0, 10, int(n_samples * 0.05))
    y_hy = a * X_hy + b + np.random.normal(0, np.sqrt(var) * 10, len(X_hy))
    
    X_both = np.random.uniform(15, 20, int(n_samples * 0.05))
    y_both = a * X_both + b + np.random.normal(0, np.sqrt(var) * 10, len(X_both))
    
    X_all = np.concatenate([X_reg, X_hx, X_hy, X_both])
    y_all = np.concatenate([y_reg, y_hx, y_hy, y_both])
    
    X_mat = np.vstack((X_all, np.ones_like(X_all))).T
    lev = leverage_scores(X_mat)
    
    ax.scatter(X_all, y_all, c=lev, cmap='viridis')
    
    idx_max = np.argsort(lev)[-5:]
    ax.scatter(X_all[idx_max], y_all[idx_max], edgecolor='red', facecolor='none', s=100)

plt.show()

fig = plt.figure(figsize=(15, 5))
for i, var in enumerate(variances):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    X1_reg = np.random.uniform(0, 10, int(n_samples * 0.85))
    X2_reg = np.random.uniform(0, 10, int(n_samples * 0.85))
    y_reg = a * X1_reg + b * X2_reg + c_val + np.random.normal(0, np.sqrt(var), len(X1_reg))
    
    X1_hx = np.random.uniform(15, 20, int(n_samples * 0.05))
    X2_hx = np.random.uniform(15, 20, int(n_samples * 0.05))
    y_hx = a * X1_hx + b * X2_hx + c_val + np.random.normal(0, np.sqrt(var), len(X1_hx))
    
    X1_hy = np.random.uniform(0, 10, int(n_samples * 0.05))
    X2_hy = np.random.uniform(0, 10, int(n_samples * 0.05))
    y_hy = a * X1_hy + b * X2_hy + c_val + np.random.normal(0, np.sqrt(var) * 10, len(X1_hy))
    
    X1_both = np.random.uniform(15, 20, int(n_samples * 0.05))
    X2_both = np.random.uniform(15, 20, int(n_samples * 0.05))
    y_both = a * X1_both + b * X2_both + c_val + np.random.normal(0, np.sqrt(var) * 10, len(X1_both))
    
    X1_all = np.concatenate([X1_reg, X1_hx, X1_hy, X1_both])
    X2_all = np.concatenate([X2_reg, X2_hx, X2_hy, X2_both])
    y_all = np.concatenate([y_reg, y_hx, y_hy, y_both])
    
    X_mat = np.vstack((X1_all, X2_all, np.ones_like(X1_all))).T
    lev = leverage_scores(X_mat)
    
    ax.scatter(X1_all, X2_all, y_all, c=lev, cmap='viridis')
    
    idx_max = np.argsort(lev)[-5:]
    ax.scatter(X1_all[idx_max], X2_all[idx_max], y_all[idx_max], edgecolor='red', facecolor='none', s=100)

plt.show()