import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data

from tqdm import tqdm
import torch_geometric.transforms as T
from pathlib import Path
import json
import numpy as np
from utils import *
import os
from config import *
from utils import *


class SessionDataset(InMemoryDataset):
    def __init__(self, root, class_labels, exclude_classes=None, transform=None, pre_transform=None, pre_filter=None):
        self.exclude_classes = exclude_classes
        self.class_labels = class_labels
        super(SessionDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root
                
        self.new_class_labels = []

    @property
    def raw_file_names(self):
        raw_files = list(Path(self.root + '\\raw').rglob('*.json'))
        raw_files = [item.name for item in raw_files]
        return raw_files

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):

        if self.exclude_classes is not None and len(self.exclude_classes) > 0:
            self.new_class_labels = [label for label in self.class_labels if label not in self.exclude_classes]
        else:
            self.new_class_labels = self.class_labels   


        pbar = tqdm(total=len(self.raw_paths), desc='Files Done: ')

        data_list = []
        for file_number, raw_file in enumerate(self.raw_paths):

            with open(raw_file, 'r') as file_handle:
                json_data = json.load(file_handle)
                
                features = json_data["features"]
                edge_indices = json_data["edge_indices"]
                class_label = json_data["class"]

                class_vector = np.zeros(len(self.new_class_labels), dtype=int)
                index = self.new_class_labels.index(class_label)
                class_vector[index] = 1

                edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
                x = torch.tensor(features, dtype=torch.float)
                y = torch.tensor(np.array([class_vector], dtype=np.float32), dtype=torch.float)
                graph = Data(x=x, edge_index=edge_index, y=y)

                data_list.append(graph)

            pbar.update(1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @classmethod
    def get_random_session(cls, num_sessions=1):
        sessions = {}
        
        x = torch.rand((10,1500))
        edge_indices = [[0,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9]]
        y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data = Data(x=x, edge_indices=edge_indices, y=y)

        print(x)

        return sessions
