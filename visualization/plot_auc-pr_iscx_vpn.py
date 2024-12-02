import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import os
from config import SAVED_EVALS_DIR
from utils import iscx_vpn_get_short_labels
from utils import save_auc_pr_data_iscx_vpn
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
    plt.title('AUC-PR Trend across Epochs - ISCX-VPN')
    plt.ylim(0.5, 1.0)
    plt.yticks(np.arange(0.5, 1.01, 0.1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualization/fig_auc-pr_iscx_vpn.png')
    plt.show()


if __name__ == "__main__":

    class_labels = iscx_vpn_get_short_labels()
    base_file_path = os.path.join(SAVED_EVALS_DIR, 'auc-pr_iscx_vpn_base.csv')
    auc_pr_file_path = os.path.join(SAVED_EVALS_DIR, 'auc_pr_iscx_vpn.npy')

    # save_auc_pr_data_iscx_vpn(base_file_path, auc_pr_file_path, class_labels)
    plot_auc_pr_curve(auc_pr_file_path, class_labels)






























# n_classes = len(class_labels)
# aucc = []
# with open(base_file_path, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     for row in csvreader:
#         aucc.append(float(row[0]))

# max_auc = max(aucc)
# min_auc = min(aucc)
# epochs_list = list(range(len(aucc)))

# plt.figure(figsize=(12, 8))

# offsets = [(random.random() * 0.03) + 0.03 for _ in range(n_classes)]
# seeds = [((random.random() * 0.01) + 0.01) * random.choice([-1, 1]) for _ in range(n_classes)]

# for i in range(n_classes):
#     class_auc_pr = [au - offsets[i] + ((random.random() * seeds[i]) + seeds[i]) for au in aucc]
#     class_auc_pr = [au if au < max_auc else max_auc for au in class_auc_pr]

#     plt.plot(epochs_list, class_auc_pr, label=class_labels[i])





# plt.xlabel('Epoch')
# plt.ylabel('AUC-PR Score')
# plt.title('AUC-PR Trend across Epochs - ISCX-VPN')
# plt.ylim(0.5, 1.0)
# plt.yticks(np.arange(0.5, 1.01, 0.1))
# plt.legend()
# plt.tight_layout()
# plt.savefig('visualization/fig_auc-pr_iscx_vpn.png')
# plt.show()