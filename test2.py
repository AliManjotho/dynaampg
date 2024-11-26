from session_dataset import SessionDataset
from utils import network_traffic_get_unique_labels, realtime_get_unique_labels
from config import NETWORK_TRAFFIC_DATASET_DIR, REALTIME_DATASET_DIR

dataset = SessionDataset(root=NETWORK_TRAFFIC_DATASET_DIR, class_labels=network_traffic_get_unique_labels())
dataset = SessionDataset(root=REALTIME_DATASET_DIR, class_labels=realtime_get_unique_labels()) 

