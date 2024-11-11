import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})

font_size = 20

# File names for the confusion matrices
csv_files = ['saved_evals/ISCX-VPN.csv', 'saved_evals/VNAT.csv', 'saved_evals/ISCX-Tor.csv']

# Plot three confusion matrices in a single row
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

for index, csv_file in enumerate(csv_files):
    # Read the confusion matrix and class labels from the CSV file
    df = pd.read_csv(csv_file)
    cm_percentage = df.values.astype(float)
    class_labels = df.columns.tolist()

    ax = axes[index]

    im = ax.imshow(cm_percentage, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size + 2)

    # Add class labels
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_labels, rotation=90, ha="right", fontsize=font_size + 2)
    ax.set_yticklabels(class_labels, fontsize=font_size + 2)

    # Add text annotations
    thresh = cm_percentage.max() / 2.

    if index == 0:
        value_font = font_size - 5
    else:
        value_font = font_size

    for i, j in np.ndindex(cm_percentage.shape):
        ax.text(j, i, f"{cm_percentage[i, j]:.3f}%",
                ha="center", va="center",
                color="white" if cm_percentage[i, j] > thresh else "black",
                fontsize=value_font)

    ax.set_xlabel('Predicted Label', fontsize=font_size + 2)
    ax.set_ylabel('True Label', fontsize=font_size + 2)
    ax.set_title(f'Confusion Matrix for {csv_file[:-4]}', fontsize=font_size + 2)

    # Draw outline on four sides of the whole plot
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

plt.tight_layout()
plt.savefig('fig_confusion_matrices.png')
plt.show()
