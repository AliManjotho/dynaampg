import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from angular_loss import ArcFace

class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=6, C=3, model_state_path = None):
        super(GraphTransformerEncoder, self).__init__()

        self.features = {}

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.C = C

        self.gtconv1 = TransformerConv(input_dim, hidden_dim, heads=num_heads, concat=False)
        self.gtconv2 = TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gtconv3 = TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gtconv4 = TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.gtconv5 = TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.3)

        self.arcface = ArcFace(num_features=hidden_dim, num_classes=output_dim)

        # self.softmax = nn.Softmax(dim=1)

        if model_state_path is not None:
            self.load(model_state_path)

    def forward(self, data, return_attention_weights=True):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = torch.relu(self.gtconv1(x, edge_index))
        self.features['gtconv1'] = x
        x = torch.relu(self.gtconv2(x, edge_index))
        self.features['gtconv2'] = x
        x = torch.relu(self.gtconv3(x, edge_index))
        self.features['gtconv3'] = x
        x = torch.relu(self.gtconv4(x, edge_index))
        self.features['gtconv4'] = x
        x = torch.relu(self.gtconv5(x, edge_index))
        self.features['gtconv5'] = x

        # Global mean pooling
        x = global_mean_pool(x, batch)

        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)


        logit = self.arcface(x)

        # pred = self.softmax(logit)

        return logit
    

    def load(self, model_state_path):
        self.load_state_dict(torch.load(model_state_path, weights_only=True))
        self.eval()


    def get_features(self):
        return self.features
    

    @torch.no_grad()
    def infer(self, session, device):
        self.to(device)
        self.eval()
        inputs = session.to(device)
        outputs = self(inputs)

        return outputs
    

    def get_weights(self):
        return {
            'gtconv1': self.gtconv1,
            'gtconv2': self.gtconv2,
            'gtconv3': self.gtconv3,
            'gtconv4': self.gtconv4,
            'gtconv5': self.gtconv5,
        }



