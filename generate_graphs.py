import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from session_dataset import SessionDataset
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from config import *
from utils import *


if __name__=='__main__':

    roots = [ISCX_VPN_DATASET_DIR, VNAT_DATASET_DIR]
    labels = [iscx_vpn_get_unique_labels(), vnat_get_unique_labels()]

    for root in roots:
        if not os.path.isdir(root + '\\raw'):
            os.mkdir(root + '\\raw')
        if not os.path.isdir(root + '\\processed'):
            os.mkdir(root + '\\processed')
        
        cmd = "move /Y " + root + "\\*.json " + root + '\\raw'
        os.system(cmd)

    iscx_dataset = SessionDataset(root=roots[0], class_labels=labels[0])
    vnat_dataset = SessionDataset(root=roots[1], class_labels=labels[1])

    print('ALL DONE!!!!!')
