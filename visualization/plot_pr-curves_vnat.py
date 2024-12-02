import sys
import os

from altair import CsvDataFormat
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import torch
from utils import vnat_get_short_labels, vnat_get_unique_labels
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import csv
from utils import save_pr_auc_vnat


pr_csv_file_path = os.path.join(SAVED_EVALS_DIR, 'pr_vnat.csv')
ap_csv_file_path = os.path.join(SAVED_EVALS_DIR, 'ap_vnat.csv')
class_labels = vnat_get_short_labels()
n_classes = len(class_labels)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 16
})




def plot_pr_curve():
    # Read precision and recall from the CSV file
    precision_recall_data = []
    with open(pr_csv_file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header
        for row in reader:
            precision_recall_data.append(row)

    # Extract precision and recall for each class
    precision = {}
    recall = {}
    average_precision = []

    for i in range(n_classes):
        precision[i] = [float(row[2*i]) if row[2*i] else None for row in precision_recall_data]
        recall[i] = [float(row[2*i+1]) if row[2*i+1] else None for row in precision_recall_data]

    with open(ap_csv_file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            average_precision.append(float(row[0]))


    plt.figure(figsize=(10, 8))
    # Add gray dashed horizontal line at y=1
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=1.0)
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'{class_labels[i]} (AP={average_precision[i]:.2f})')

    # Add plot details
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest) - ISCX-VPN")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig('visualization/fig_pr_curve_iscx_vpn.png')
    plt.show()


    


if __name__ == "__main__":

    # save_pr_auc_vnat(pr_csv_file_path, ap_csv_file_path, class_labels, n_classes)
    plot_pr_curve()
