import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from baseline import BaselineClassifier
from baseline_dataset import BaselineDataset
from config import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from utils import iscx_vpn_get_unique_labels

def plot_confusion_matrix(model, test_loader, num_classes, device, class_names, class_counts):
    all_true_labels = []
    all_predicted_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm.tqdm(test_loader, desc="Making predictions"):
            features = features.to(device)
            # Get predictions from your model
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            # Convert one-hot encoded labels back to class indices
            true_labels = torch.argmax(labels, dim=1)
            
            # Store true labels and predicted labels
            all_true_labels.extend(true_labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())  # Use predicted labels
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with percentage values
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix (Percentages)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_pr_curves(model, test_loader, num_classes, device, class_names, class_counts):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Plot PR curve for each class
    plt.figure(figsize=(10, 8))
    
    precisions = []
    recalls = []
    pr_aucs = []

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        pr_auc = auc(recall, precision)
        precisions.append(precision)
        recalls.append(recall)
        pr_aucs.append(pr_auc)

    # Sort class names and PR AUCs together in descending order of AUCs
    sorted_indices = np.argsort(pr_aucs)[::-1]  # Sort in descending order
    class_names = [class_names[i] for i in sorted_indices]
    pr_aucs = [pr_aucs[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]

    # Plot curves in sorted order
    for i in range(len(sorted_indices)):
        plt.plot(recalls[i], precisions[i], label=f'{class_names[i]} (AUC = {pr_aucs[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for All Classes')
    plt.legend()
    plt.grid(True)
    plt.savefig('pr_curves.png')
    plt.show()

if __name__ == "__main__":

    LOAD_MODEL = os.path.join(SAVED_MODELS_DIR, 'mlp_classifier_9.pth')
    batch_size = 32
    hidden_size = 128

    dataset = BaselineDataset(os.path.join(ISCX_VPN_DATASET_DIR, "raw"), iscx_vpn_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders()    
    
    # Get input size from first sample
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(LOAD_MODEL))
    model = model.to(device)    
 

    # Get class names from the encoder
    class_names =  list(dataset.get_instances_per_class().keys())
    class_counts = dataset.get_instances_per_class()


    # Generate and save PR curves
    plot_pr_curves(model, test_loader, num_classes, device, class_names, class_counts)    
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, num_classes, device, class_names, class_counts)







    
