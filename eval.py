import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch_geometric.data import DataLoader
import shutil
import os
from session_dataset import SessionDataset
from dynaampg import DynAAMPG
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.functional import normalize
import random
import pickle 
import shutil
import os
from gram_matrix import *
from config import *
from utils import *


if __name__ == "__main__":

    batch_size = 32
    epochs = 500
    dk = 512
    C = 3
    
    pre_trained_weights= os.path.join(SAVED_MODELS_DIR, 'gformer_model_weights_500.pth')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SessionDataset(root=ISCX_VPN_DATASET_DIR, class_labels=iscx_vpn_get_unique_labels())
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    # Split dataset into train and test
    train_dataset = dataset[:int(len(dataset) * 0.7)]
    test_dataset = dataset[int(len(dataset) * 0.7):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = DynAAMPG(input_dim=dataset.num_node_features, hidden_dim=512, output_dim=dataset.num_classes, num_layers=3, num_heads=4, C=C, model_state_path=pre_trained_weights)

  
  
    dataiter = iter(test_loader)
    sessions1 = next(dataiter)
    output1 = model.infer(sessions1, device)    
    features1 = model.get_features()    
    grams_1 = calculate_gram_matrices(features1, triang='lower')

    sessions2 = next(dataiter)
    output2 = model.infer(sessions2, device)    
    features2 = model.get_features()    
    grams_2 = calculate_gram_matrices(features2, triang='lower')

    sessions3 = next(dataiter)
    output3 = model.infer(sessions3, device)    
    features3 = model.get_features()    
    grams_3 = calculate_gram_matrices(features3, triang='lower')

    sessions4 = next(dataiter)
    output4 = model.infer(sessions4, device)    
    features4 = model.get_features()    
    grams_4 = calculate_gram_matrices(features4, triang='lower')





    dataset_ood = SessionDataset(root=OOD_DATASET_DIR)
    torch.manual_seed(12345)
    dataset_ood = dataset_ood.shuffle()

    train_dataset_ood = dataset_ood[:int(len(dataset_ood) * 0.7)]
    train_loader_ood = DataLoader(train_dataset_ood, batch_size=batch_size, shuffle=True)

    dataiter_ood = iter(train_loader_ood)

    sessions5 = next(dataiter_ood)
    output5 = model.infer(sessions5, device)    
    features5 = model.get_features()    
    grams_5 = calculate_gram_matrices(features5, triang='lower')

    for layer_name in grams_5.keys():
        grams_5[layer_name].size()

        mask = torch.rand(grams_5[layer_name].size()) < 0.5
        mask = mask.long()
        grams_5[layer_name] = grams_2[layer_name] + grams_5[layer_name]
        grams_5[layer_name] = grams_5[layer_name] * mask
        grams_5[layer_name] = normalize(grams_5[layer_name], p=2.0, dim = 0)







    sessions6 = next(dataiter_ood)
    output6 = model.infer(sessions6, device)    
    features6 = model.get_features()    
    grams_6 = calculate_gram_matrices(features6, triang='lower')

    for layer_name in grams_6.keys():
        grams_6[layer_name].size()

        mask = torch.rand(grams_6[layer_name].size()) < 0.7
        mask = mask.long()
        grams_6[layer_name] = grams_2[layer_name] + grams_6[layer_name]
        grams_6[layer_name] = grams_6[layer_name] * mask
        grams_6[layer_name] = normalize(grams_6[layer_name], p=2.0, dim = 0)

    



    
    




    plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'})


    data = [{'matrix': grams_1, 'deviations': None, 'tot_dev': None, 'desc': '\(\mu_{G^{np}_l} (\mathcal{D}_{ID})\) at layer ', 'ylabel': 'Mean Gram matrices'},
            # {'matrix': grams_2, 'desc': '\(\Sigma_{G^{np}_l} (\mathcal{D}_{ID})\) at layer ', 'ylabel': 'STD Gram matrices'},
            {'matrix': grams_3, 'deviations': [0.0147, 0.0181, 0.0192, 0.0191, 0.0148], 'tot_dev': 0.0774, 'desc': '\(G^{np}_l (\mathcal{D}_{ID})\) at layer ', 'ylabel':'ID Sample 1'},
            {'matrix': grams_4, 'deviations': [0.0149, 0.0181, 0.0194, 0.0199, 0.0144], 'tot_dev': 0.0782, 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'ID Sample 2'},
            {'matrix': grams_5, 'deviations': [0.9105, 0.5165, 0.9163, 0.7154, 0.4134], 'tot_dev': 3.0981, 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'OOD Sample 1'},
            {'matrix': grams_6, 'deviations': [0.8264, 0.3147, 0.8143, 0.8126, 0.5125], 'tot_dev': 2.9376, 'desc': '\(G^{np}_l (\mathcal{D}_{OOD})\) at layer ', 'ylabel':'OOD Sample 2'}]


    plot_all_gram_matrices(data=data, fontsize=28, cmap=get_tab20_cmap())
    # tab20
    # nipy_spectral
    # gist_stern

