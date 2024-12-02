import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import os
from config import SAVED_EVALS_DIR
from utils import vnat_get_short_labels
from utils import save_auc_pr_data_vnat
import numpy as np



plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 16
})

def plot_auc_pr_curve(auc_pr_file_path, class_labels):
    auccs = np.load(auc_pr_file_path)
    
    epochs_list = list(range(auccs.shape[1]))
    
    plt.figure(figsize=(12, 8))   

    for i, aucc in enumerate(auccs):
        plt.plot(epochs_list, aucc, label=class_labels[i])

    plt.xlabel('Epoch')
    plt.ylabel('AUC-PR Score')
    plt.title('AUC-PR Trend across Epochs - VNAT')
    plt.ylim(0.5, 1.0)
    plt.yticks(np.arange(0.5, 1.01, 0.1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualization/fig_auc-pr_vnat.png')
    plt.show()


if __name__ == "__main__":

    class_labels = vnat_get_short_labels()
    base_file_path = os.path.join(SAVED_EVALS_DIR, 'auc-pr_vnat_base.csv')
    auc_pr_file_path = os.path.join(SAVED_EVALS_DIR, 'auc_pr_vnat.npy')

    # save_auc_pr_data_vnat(base_file_path, auc_pr_file_path, class_labels)
    plot_auc_pr_curve(auc_pr_file_path, class_labels)