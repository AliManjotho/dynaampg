import matplotlib.pyplot as plt
import numpy as np
from config import *
import os
import json


root = ISCX_VPN_DATASET_DIR
exclude_classes = ['email', 'voip']
class_labels = []



with open(os.path.join(root, 'raw/meta.json'), 'r') as file_handle:
    meta_json_data = json.load(file_handle)
    class_labels = meta_json_data['class_labels']

new_class_labels = [label for label in class_labels if label not in exclude_classes]
num_classes = len(new_class_labels)

class_vector = np.zeros(num_classes, dtype=int)

class_label = 'vpn_chat'

index = new_class_labels.index(class_label)
class_vector[index] = 1



for class_label in new_class_labels:
    index = new_class_labels.index(class_label)
    class_vector = np.zeros(num_classes, dtype=int)
    class_vector[index] = 1
    print(class_label, class_vector)