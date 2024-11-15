import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
import shutil
from session_dataset import SessionDataset
from dynaampg import GraphTransformerEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.functional import normalize
import random
import pickle 
import shutil
import os
from gram_matrix import *
from config import *

def get_mean_grams(pre_trained_weights):
    batch_size = 32
    epochs = 500
    dk = 512
    C = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR)
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    test_dataset = dataset[int(len(dataset) * 0.7):]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphTransformerEncoder(input_dim=dataset.num_node_features, hidden_dim=512, output_dim=dataset.num_classes, num_layers=3, num_heads=4, C=C, model_state_path=pre_trained_weights)

    dataiter = iter(test_loader)
    session = next(dataiter)
    output = model.infer(session, device)    
    features = model.get_features()    
    means_grams = calculate_gram_matrices(features, triang='lower')

    return means_grams

def get_id_grams(num_samples, pre_trained_weights):
    grams = []

    batch_size = 32
    epochs = 500
    dk = 512
    C = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR)
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    test_dataset = dataset[int(len(dataset) * 0.7):]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphTransformerEncoder(input_dim=dataset.num_node_features, hidden_dim=512, output_dim=dataset.num_classes, num_layers=3, num_heads=4, C=C, model_state_path=pre_trained_weights)

    dataiter = iter(test_loader)
    s = next(dataiter)
    for i in range(num_samples):
        session = next(dataiter)
        output = model.infer(session, device)    
        features = model.get_features()    
        gram = calculate_gram_matrices(features, triang='lower')
        grams.append(gram)

    return grams

def get_ood_grams(num_samples, masks, ref_gram, pre_trained_weights):
    grams = []

    batch_size = 32
    epochs = 500
    dk = 512
    C = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SessionDataset(root=OOD_DATASET_DIR)
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    test_dataset = dataset[int(len(dataset) * 0.7):]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GraphTransformerEncoder(input_dim=dataset.num_node_features, hidden_dim=512, output_dim=dataset.num_classes, num_layers=3, num_heads=4, C=C, model_state_path=pre_trained_weights)

    dataiter = iter(test_loader)

    for i in range(num_samples):
        session = next(dataiter)
        output = model.infer(session, device)    
        features = model.get_features()    
        gram = calculate_gram_matrices(features, triang='lower')

        for layer_name in gram.keys():
            gram[layer_name].size()

            mask = torch.rand(gram[layer_name].size()) < masks[i]
            mask = mask.long()
            gram[layer_name] = ref_gram[layer_name] + gram[layer_name]
            gram[layer_name] = gram[layer_name] * mask
            gram[layer_name] = normalize(gram[layer_name], p=2.0, dim = 0)

        grams.append(gram)

    return grams


if __name__ == "__main__":

    pre_trained_weights = os.path.join(SAVED_MODELS_DIR, 'gformer_model_weights_500.pth')

    mean_grams = get_mean_grams(pre_trained_weights)
    id_grams = get_id_grams(2, pre_trained_weights)
    ood_grams = get_ood_grams(2, [0.5, 0.7], id_grams[0], pre_trained_weights)


    plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'})


    devs_file_path = os.path.join(SAVED_DEVS_DIR, 'dev_data.json')
    devs_data = load_dev_data(devs_file_path)

    data = [{'matrix': mean_grams, 'deviations': None, 'tot_dev': None, 'desc': '\(\mu_{G^{np}_l} (\mathcal{D}_{ID})\) at layer ', 'ylabel': 'Mean Gram matrices'},
            {'matrix': id_grams[0], 'deviations': devs_data[0]['layer_devs'], 'tot_dev':devs_data[0]['tot_dev'], 'desc': '\(G^{np}_l (\mathcal{D}_{ID})\) at layer ', 'ylabel':'ID Sample 1'},
            {'matrix': id_grams[1], 'deviations': devs_data[1]['layer_devs'], 'tot_dev':devs_data[1]['tot_dev'], 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'ID Sample 2'},
            {'matrix': ood_grams[0], 'deviations': devs_data[2]['layer_devs'], 'tot_dev':devs_data[2]['tot_dev'], 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'OOD Sample 1'},
            {'matrix': ood_grams[1], 'deviations': devs_data[3]['layer_devs'], 'tot_dev':devs_data[3]['tot_dev'], 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'OOD Sample 2'}]


    plot_all_gram_matrices(data=data, fontsize=28, cmap=get_tab20_cmap())
    # tab20
    # nipy_spectral
    # gist_stern
