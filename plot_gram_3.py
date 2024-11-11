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


def get_matrices(height, width):
    feature = torch.randn(height, width).to(device)
    (h, w) = feature.size()
    gram = torch.mm(feature, feature.transpose(0, 1))
    gram = (gram / (h * w))
    gram_tri = torch.tril(gram)
    return feature, gram, gram_tri


# Calculate and plot Gram matrices for L layers
L = 5
height = 20
width = 20

fig, axes = plt.subplots(3, L, figsize=(15, 3*5))
for i in range(L):
    feature, gram, gram_tri = get_matrices(height, width)

    ax = axes[0, i]
    ax.imshow(feature.detach().cpu().numpy(), cmap='viridis')
    ax.set_title(f'Layer {i+1}')
    ax.axis('off')

    ax = axes[1, i]
    ax.imshow(gram.detach().cpu().numpy(), cmap='viridis')
    ax.axis('off')

    ax = axes[2, i]
    ax.imshow(gram_tri.detach().cpu().numpy(), cmap='viridis')
    ax.axis('off')

plt.suptitle('Gram Matrices for Features Across Layers')
# plt.colorbar(axes[0, 0].images[0], ax=axes, orientation='vertical', fraction=.1)
plt.show()
