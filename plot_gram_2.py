import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained model (e.g., VGG19)
model = models.vgg19(pretrained=True).features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Function to load and preprocess an image
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to calculate Gram matrix
def gram_matrix(features):
    (b, c, h, w) = features.size()  # Batch, Channels, Height, Width
    features = features.view(b, c, h * w)  # Flatten the spatial dimensions
    gram = torch.bmm(features, features.transpose(1, 2))  # Batch matrix-matrix product
    gram = torch.tril(gram)
    return gram / (c * h * w)

# Generate random feature maps for demonstration
def generate_random_features(batch_size, channels, height, width):
    return torch.randn(batch_size, channels, height, width).to(device)

# Calculate and plot Gram matrices for L layers
L = 5
batch_size = 1
channels = 64
height = 10
width = 10

fig, axes = plt.subplots(1, L, figsize=(15, 5))
for i in range(L):
    random_features = generate_random_features(batch_size, channels, height, width)
    gram_random = gram_matrix(random_features).squeeze().detach().cpu().numpy()
    
    # random_features = random_features.squeeze().detach().cpu().numpy()
    ax = axes[i]
    ax.imshow(gram_random, cmap='viridis')
    ax.set_title(f'Layer {i+1}')
    ax.axis('off')

plt.suptitle('Gram Matrices for Random Features Across Layers')
plt.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=.1)
plt.show()
