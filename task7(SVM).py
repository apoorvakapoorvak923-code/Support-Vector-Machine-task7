# src/svm_task.py
# Complete SVM demo: load data, prepare 2D projection, train linear/RBF SVMs,
# tune hyperparameters with GridSearchCV, evaluate with cross-validation,
# plot decision boundaries and save results.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ----------------------
# 1) Create outputs folder
# ----------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# 2) Load dataset
# ----------------------
data = load_breast_cancer()
X = data.data                      # shape (n_samples, n_features)
y = data.target                    # 0/1 labels
feature_names = data.feature_names

# create a DataFrame (optional - good for exploration)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

# ----------------------
# 3) Create a 2D version for visualization
#    Option A: choose two features (uncomment to use)
# ----------------------
# feature_indices = [0, 1]  # mean radius, mean texture (example)
# X_2d = X[:, feature_indices]

# ----------------------
# Option B (recommended): use PCA to project to 2D (keeps most variance)
# ----------------------
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
print("Explained variance ratio by 2 PCA components:", pca.explained_variance_ratio_.sum())

# ----------------------
# 4) Split into train/test for evaluation
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 5) Scale features (important for SVM)
# ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler (so you can reuse it later)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

# ----------------------
# 6) Set up hyperparameter grids and GridSearchCV
# ----------------------
# Linear SVM: tune C
param_grid_linear = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']}

# RBF SVM: tune C and gamma
param_grid_rbf = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Grid searches
grid_linear = GridSearchCV(SVC(), param_grid_linear, cv=5, scoring='accuracy', n_jobs=-1)
grid_rbf = GridSearchCV(SVC(), param_grid_rbf, cv=5, scoring='accuracy', n_jobs=-1)

print("Starting GridSearch for linear SVM...")
grid_linear.fit(X_train_scaled, y_train)
print("Best linear params:", grid_linear.best_params_)

print("Starting GridSearch for RBF SVM...")
grid_rbf.fit(X_train_scaled, y_train)
print("Best RBF params:", grid_rbf.best_params_)

best_linear = grid_linear.best_estimator_
best_rbf = grid_rbf.best_estimator_

# Save the models
joblib.dump(best_linear, os.path.join(OUT_DIR, "svm_linear_best.joblib"))
joblib.dump(best_rbf, os.path.join(OUT_DIR, "svm_rbf_best.joblib"))

# ----------------------
# 7) Evaluation on test set
# ----------------------
def evaluate_model(model, Xs, ys, name):
    preds = model.predict(Xs)
    acc = accuracy_score(ys, preds)
    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(classification_report(ys, preds))
    cm = confusion_matrix(ys, preds)
    print(f"{name} Confusion Matrix:\n{cm}")
    return acc, cm

evaluate_model(best_linear, X_test_scaled, y_test, "Linear SVM")
evaluate_model(best_rbf, X_test_scaled, y_test, "RBF SVM")

# ----------------------
# 8) Cross-validation scores (on full scaled 2D data)
# ----------------------
X_scaled_full = scaler.transform(X_2d)
cv_scores_lin = cross_val_score(best_linear, X_scaled_full, y, cv=5, scoring='accuracy', n_jobs=-1)
cv_scores_rbf = cross_val_score(best_rbf, X_scaled_full, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCV accuracy linear:", cv_scores_lin.mean(), "±", cv_scores_lin.std())
print("CV accuracy RBF:", cv_scores_rbf.mean(), "±", cv_scores_rbf.std())

# ----------------------
# 9) Plot decision boundaries (works because we use 2D data)
# ----------------------
def plot_decision_boundary(model, X_plot, y_plot, title, filename):
    # X_plot should be unscaled or scaled consistently with model training
    x_min, x_max = X_plot[:, 0].min() - 1.0, X_plot[:, 0].max() + 1.0
    y_min, y_max = X_plot[:, 1].min() - 1.0, X_plot[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.2)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, s=30, edgecolors='k')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0], labels=['class 0', 'class 1'])
    outpath = os.path.join(OUT_DIR, filename)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved plot:", outpath)

# For plotting, the model expects scaled input (we trained on scaled data)
plot_decision_boundary(best_linear, X_scaled_full, y, "Linear SVM decision boundary (PCA 2D)", "linear_decision_boundary.png")
plot_decision_boundary(best_rbf, X_scaled_full, y, "RBF SVM decision boundary (PCA 2D)", "rbf_decision_boundary.png")

print("\nAll outputs (models, scaler, plots) are in the 'outputs' folder.")
