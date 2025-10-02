# Support-Vector-Machine-task7
# Task 7 - Support Vector Machine (SVM) Classification

This project demonstrates the use of **Support Vector Machines (SVM)** for binary classification using the **Breast Cancer dataset** from scikit-learn.  
The task involves:
- Loading and preprocessing data
- Reducing dimensions for visualization
- Training **Linear** and **RBF (Radial Basis Function)** SVM classifiers
- Tuning hyperparameters using **GridSearchCV**
- Evaluating model performance using accuracy, confusion matrix, and cross-validation
- Visualizing decision boundaries
- Saving trained models and scaler for future use

---
## 📂 Project Structure
svm_task/
├── src/
│ └── task7(SVM).py # Main Python script
├── outputs/ # Folder for plots & saved models (auto-created)
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── .gitignore # Ignored files/folders
---

## ⚙️ Features Implemented
- **Data Loading**: Breast Cancer dataset with 569 samples and 30 features.
- **Dimensionality Reduction**: Used PCA (2 components) to project high-dimensional data into 2D for visualization.
- **Preprocessing**: Features scaled using `StandardScaler` for better SVM performance.
- **Model Training**:
  - Linear SVM (kernel = linear)
  - RBF SVM (kernel = rbf)
- **Hyperparameter Tuning**:
  - Linear SVM: Grid search over `C = [0.01, 0.1, 1, 10, 100]`
  - RBF SVM: Grid search over `C = [0.1, 1, 10, 100]` and `gamma = ['scale', 'auto', 0.01, 0.1, 1]`
- **Evaluation**:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
  - Train/test accuracy
  - 5-fold cross-validation
- **Visualization**:
  - Decision boundary plots for Linear and RBF models in 2D
- **Persistence**:
  - Saved trained models (`.joblib`)
  - Saved fitted scaler

---

## 🧮 Results
- **Best Linear SVM Parameters**: `C = 10`
  - Test Accuracy: **~0.94**
  - Confusion Matrix:
    ```
    [[37   5]
     [ 2  70]]
    ```
- **Best RBF SVM Parameters**: `C = 100`, `gamma = 0.1`
  - Test Accuracy: **~0.90**
  - Confusion Matrix:
    ```
    [[37   5]
     [ 6  66]]
    ```
- **Cross-validation Accuracy**:
  - Linear SVM: **0.9227 ± 0.0178**
  - RBF SVM: **0.9244 ± 0.0133**

---

## 📊 Outputs
All results are saved in the **`outputs/` folder**:
- `linear_decision_boundary.png` – decision boundary of Linear SVM
- `rbf_decision_boundary.png` – decision boundary of RBF SVM
- `svm_linear_best.joblib` – saved Linear SVM model
- `svm_rbf_best.joblib` – saved RBF SVM model
- `scaler.joblib` – saved StandardScaler

---

## 🛠️ Installation

### 1. Clone the repository
```bash


cd <repo-name>
