import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# --- Plot 1: Class Imbalance --- #
def plot_class_imbalance():
    # Create synthetic dataset with imbalance
    n_samples = 1000
    n_features = 2
    class_0 = np.random.randn(n_samples // 10, n_features)  # Majority class (90%)
    class_1 = np.random.randn(n_samples * 9 // 10, n_features) + 3  # Minority class (10%)
    X = np.vstack([class_0, class_1])
    y = np.array([0] * (n_samples // 10) + [1] * (n_samples * 9 // 10))
    
    # Updated histogram plotting
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=2, color='blue', edgecolor='black')
    plt.xticks([0, 1], ['Class 0 (Majority)', 'Class 1 (Minority)'])
    plt.ylabel('Number of Samples')
    plt.title('Class Imbalance in Dataset')
    plt.show()

# --- Plot 2: OOD Detection --- #
def plot_ood_detection():
    # Create synthetic dataset for in-distribution data with corrected parameters
    X_in, y_in = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2,
        n_redundant=0,  # Added this parameter
        n_repeated=0,   # Added this parameter
        n_classes=2, 
        random_state=42
    )
    
    # Create synthetic out-of-distribution (OOD) data
    X_ood = np.random.randn(100, 2) * 3 + 10  # OOD data far from in-distribution data

    # Plotting in-distribution data and OOD samples
    plt.figure(figsize=(8, 6))
    plt.scatter(X_in[:, 0], X_in[:, 1], c=y_in, cmap='coolwarm', label='In-distribution', alpha=0.6)
    plt.scatter(X_ood[:, 0], X_ood[:, 1], color='gray', label='Out-of-distribution', alpha=0.6)
    plt.title('Out-of-Distribution (OOD) Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# --- Plot 3: Combined Figure --- #
def plot_combined():
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Left Plot (Class Imbalance) ---
    n_samples = 1000
    n_features = 2
    class_0 = np.random.randn(n_samples // 10, n_features)  # Majority class (90%)
    class_1 = np.random.randn(n_samples * 9 // 10, n_features) + 3  # Minority class (10%)
    X = np.vstack([class_0, class_1])
    y = np.array([0] * (n_samples // 10) + [1] * (n_samples * 9 // 10))
    
    ax[0].hist(y, bins=2, color='blue', edgecolor='black')
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Class 0 (Majority)', 'Class 1 (Minority)'])
    ax[0].set_ylabel('Number of Samples')
    ax[0].set_title('Class Imbalance in Dataset')
    
    # --- Right Plot (OOD Detection) ---
    X_in, y_in = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2,
        n_redundant=0,  # Added this parameter
        n_repeated=0,   # Added this parameter
        n_classes=2, 
        random_state=42
    )
    X_ood = np.random.randn(100, 2) * 3 + 10  # OOD data far from in-distribution data
    
    ax[1].scatter(X_in[:, 0], X_in[:, 1], c=y_in, cmap='coolwarm', label='In-distribution', alpha=0.6)
    ax[1].scatter(X_ood[:, 0], X_ood[:, 1], color='gray', label='Out-of-distribution', alpha=0.6)
    ax[1].set_title('Out-of-Distribution (OOD) Detection')
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 2')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

# --- Call functions to plot the figures --- #
plot_class_imbalance()
plot_ood_detection()
plot_combined()
