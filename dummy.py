import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib
from session_dataset import SessionDataset
import torch


x = torch.rand((3,5))

tensor = torch.rand((3, 5)) < 0.7
tensor = tensor.long()

x = x * tensor
print(x)
print(tensor)