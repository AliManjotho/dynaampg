import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils import iscx_tor_get_short_labels
import matplotlib.pyplot as plt
import csv
from utils import save_pr_iscx_tor


pr_csv_file_path = os.path.join(SAVED_EVALS_DIR, 'pr_iscx_tor.csv')
ap_csv_file_path = os.path.join(SAVED_EVALS_DIR, 'ap_iscx_tor.csv')
class_labels = iscx_tor_get_short_labels()
n_classes = len(class_labels)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "font.size": 16
})




def plot_pr_curve(plot_title, file_name):

    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#808000', 
        '#008000', '#800080', '#008080', '#000080', '#FFA500', '#A52A2A', '#8A2BE2', '#5F9EA0', 
        '#D2691E', '#FF7F50', '#6495ED', '#DC143C'
    ]
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
        plt.plot(recall[i], precision[i], color=colors[i], label=f'{class_labels[i]} (AP={average_precision[i]:.2f})', linewidth=2, alpha=0.5)

    # Add plot details
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(plot_title)
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'visualization/{file_name}')
    plt.show()


    


if __name__ == "__main__":

    saved_model = os.path.join(SAVED_MODELS_DIR, 'gformer_model_weights_iscx_tor_10.pth')
    save_pr_iscx_tor(pr_csv_file_path, ap_csv_file_path, class_labels, n_classes, saved_model=saved_model)
    plot_pr_curve(plot_title='Precision-Recall Curve (One-vs-Rest) - ISCX-TOR (No-Margins)', file_name='fig_pr_curve_iscx_tor_nomargins.png')

    saved_model = os.path.join(SAVED_MODELS_DIR, 'gformer_model_weights_iscx_tor_490.pth')
    save_pr_iscx_tor(pr_csv_file_path, ap_csv_file_path, class_labels, n_classes, saved_model=saved_model)
    plot_pr_curve(plot_title='Precision-Recall Curve (One-vs-Rest) - ISCX-TOR (DynAAM)', file_name='fig_pr_curve_iscx_tor_dynaam.png')
