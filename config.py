import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
ISCX_VPN_DATASET_DIR = os.path.join(BASE_DIR, 'datasets\\ISCX_VPN')
VNAT_DATASET_DIR = os.path.join(BASE_DIR, 'datasets\\VNAT')
OOD_DATASET_DIR = os.path.join(BASE_DIR, 'datasets\\OOD')


SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
SAVED_DEVS_DIR = os.path.join(BASE_DIR, 'saved_devs')
SAVED_EVALS_DIR = os.path.join(BASE_DIR, 'saved_evals')
SAVED_MARGINS_DIR = os.path.join(BASE_DIR, 'saved_margins')

# Tensorboard log directory
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, 'runs')

# SplitCap path
SPLITCAP_DIR = os.path.join(BASE_DIR, 'SplitCap.exe')

