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
    return gram / (c * h * w)

# Hook to extract features from a specific layer
def get_features(model, layer, x):
    features = []
    def hook_fn(module, input, output):
        features.append(output)
    
    handle = model[layer].register_forward_hook(hook_fn)
    model(x)
    handle.remove()
    return features[0]

# Load the image
img_path = "path_to_your_image.jpg"  # Replace with your image path
image = load_image(img_path)

# Choose a layer index (e.g., layer 5)
layer_idx = 5

# Get the feature maps at the specified layer
features = get_features(model, layer_idx, image)

# Calculate the Gram matrix
gram = gram_matrix(features).squeeze().detach().cpu().numpy()

# Plot the Gram matrix
plt.figure(figsize=(10, 10))
plt.imshow(gram, cmap='viridis')
plt.title(f'Gram Matrix at Layer {layer_idx}')
plt.colorbar()
plt.show()
