import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader
from config import *
from session_dataset import SessionDataset
from utils import *
import numpy as np
from matplotlib.colors import ListedColormap
from config import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})




def plot_dataset_distribution(dataset, dataset_name, dataset_labels, file_name,method='PCA'):
    node_features = []
    labels = []
    labels_int = []

    for data in dataset:
        node_features.append(data.x.view(-1).cpu().numpy())
        label = (data.y[0] == 1.0).nonzero(as_tuple=False).item()
        labels.append(dataset_labels[label])
        labels_int.append(label)

    # Convert list to numpy array
    node_features = np.array(node_features)

    # Apply PCA or t-SNE for dimensionality reduction
    if method == 'PCA':
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(node_features)
    elif method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(node_features)

    # Define custom colormap with 15 distinct colors
    # custom_colors = [
    #     '#FF0000',  # Red
    #     '#00FF00',  # Green
    #     '#0000FF',  # Blue
    #     '#FFFF00',  # Yellow
    #     '#FF00FF',  # Magenta
    #     '#00FFFF',  # Cyan
    #     '#800000',  # Maroon
    #     '#808000',  # Olive
    #     '#800080',  # Purple
    #     '#008080',  # Teal
    #     '#000080',  # Navy
    #     '#FFA500',  # Orange
    #     '#A52A2A',  # Brown
    #     '#8A2BE2',  # BlueViolet
    #     '#5F9EA0',  # CadetBlue
    # ]

    # Plot the 2D reduced features
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels_int, cmap=ListedColormap(custom_colors), s=40)
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels_int, cmap="tab20", s=50)
    plt.legend(scatter.legend_elements()[0], dataset_labels)
    plt.title(f'{dataset_name} Distribution ({method})')
    plt.xlabel('Principal Component 1' if method == 'PCA' else 't-SNE Component 1')
    plt.ylabel('Principal Component 2' if method == 'PCA' else 't-SNE Component 2')
    plt.savefig(file_name, dpi=600)
    plt.show()



iscx_vpn_dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
vnat_dataset = SessionDataset(root=VNAT_DATASET_DIR, class_labels=vnat_get_unique_labels())

plot_dataset_distribution(iscx_vpn_dataset, dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_pca.png', method='PCA')
plot_dataset_distribution(iscx_vpn_dataset, dataset_name='ISCX-VPN', dataset_labels=iscx_vpn_get_unique_labels(), file_name='visualization/fig_iscx_distribution_tsne.png', method='t-SNE')  

plot_dataset_distribution(vnat_dataset, dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_pca.png', method='PCA')  
plot_dataset_distribution(vnat_dataset, dataset_name='VNAT', dataset_labels=vnat_get_unique_labels(), file_name='visualization/fig_vnat_distribution_tsne.png', method='t-SNE')  