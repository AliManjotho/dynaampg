import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from graph_transformer import GraphTransformerEncoder
from session_dataset import SessionDataset
from torch.utils.tensorboard import SummaryWriter
import shutil
import os

# Training loop
def train(train_loader, model, optimizer, criterion, device):
    model = model.to(device)
    model.train()
    total_loss = 0
    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing loop
@torch.no_grad()
def test(test_loader, model, device):
    model = model.to(device)
    model.eval()
    correct = 0
    for data in test_loader:
        data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y.argmax(dim=1)).sum())
    return correct / len(test_loader.dataset)



if __name__ == "__main__":

    batch_size = 32
    epochs = 1200
    dk = 512
    C = 3
    save_dir = "saved_models"
    tensorboard_logs = "runs"

    iscx_root = 'D:/SH/CODE/gformer/datasets/iscx'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if os.path.exists(tensorboard_logs):
        shutil.rmtree(tensorboard_logs)

  
    writer = SummaryWriter()

    dataset = SessionDataset(root=iscx_root)
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    # Split dataset into train and test
    train_dataset = dataset[:int(len(dataset) * 0.7)]
    test_dataset = dataset[int(len(dataset) * 0.7):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = GraphTransformerEncoder(input_dim=dataset.num_node_features, hidden_dim=512, output_dim=dataset.num_classes, num_layers=3, num_heads=4, C=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()


    # Main training and testing process
    max_train_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss = train(train_loader, model, optimizer, criterion, device)
        train_acc = test(train_loader, model, device)
        test_acc = test(test_loader, model, device)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        writer.add_scalars('Loss', {'Train Loss':train_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train Acc':train_acc, 'Test Acc':test_acc} , epoch)

        if train_acc > max_train_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, "gformer_model_weights_" + str(epoch) + ".pth"))
            max_train_acc = train_acc


    writer.flush()
    writer.close()