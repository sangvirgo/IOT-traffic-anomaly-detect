"""
models.py
GCN model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_IDS(nn.Module):
    """Graph Convolutional Network for Intrusion Detection"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        super(GCN_IDS, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    def predict_proba(self, data):
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.exp(logits)
        return probs
    
    def predict(self, data):
        probs = self.predict_proba(data)
        return probs.argmax(dim=1)