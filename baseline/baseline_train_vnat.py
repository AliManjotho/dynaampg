import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from baseline_dataset import BaselineDataset
from baseline_alt import BaselineAltClassifier
from config import *
from utils import vnat_get_unique_labels

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == actual).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        torch.save(model.state_dict(), f'{SAVED_MODELS_DIR}/baseline_classifier_vnat_{epoch}.pth')


if __name__ == "__main__":
    batch_size = 32

    dataset = BaselineDataset(os.path.join(VNAT_DATASET_DIR, "raw"), vnat_get_unique_labels())
    train_loader, test_loader = dataset.get_train_test_loaders(split=0.3)    
    
    # Get input size from first sample
    sample_features, sample_labels = dataset[0]
    input_size = sample_features.shape[0] * sample_features.shape[1]
    num_classes = sample_labels.shape[0]
    
    # Create and train model
    hidden_size = 128
    model = BaselineAltClassifier(input_size, hidden_size, num_classes)
    
    # Train the model
    train_model(model, train_loader, num_epochs=10)

