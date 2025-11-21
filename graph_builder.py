import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict

class IPBasedGraphBuilder:
    def __init__(self):
        self.ip_to_idx = {}
        self.idx_to_ip = {}
        self.node_counter = 0
    
    def build(self, features, labels, src_ips, dst_ips, device='cpu'):
        if src_ips is None or dst_ips is None:
            raise ValueError("No IP addresses provided!")
        
        all_ips = set(src_ips) | set(dst_ips)
        for ip in all_ips:
            if ip not in self.ip_to_idx:
                self.ip_to_idx[ip] = self.node_counter
                self.idx_to_ip[self.node_counter] = ip
                self.node_counter += 1
        
        num_nodes = len(self.ip_to_idx)
        
        node_features = np.zeros((num_nodes, features.shape[1]))
        node_counts = np.zeros(num_nodes)
        node_label_votes = defaultdict(lambda: defaultdict(int))
        
        for i, (src_ip, dst_ip) in enumerate(zip(src_ips, dst_ips)):
            src_idx = self.ip_to_idx[src_ip]
            dst_idx = self.ip_to_idx[dst_ip]
            
            node_features[src_idx] += features[i]
            node_features[dst_idx] += features[i]
            
            node_counts[src_idx] += 1
            node_counts[dst_idx] += 1
            
            node_label_votes[src_idx][labels[i]] += 1
            node_label_votes[dst_idx][labels[i]] += 1
        
        for i in range(num_nodes):
            if node_counts[i] > 0:
                node_features[i] /= node_counts[i]
        
        node_labels = np.zeros(num_nodes, dtype=int)
        for i in range(num_nodes):
            if node_label_votes[i]:
                votes = node_label_votes[i]
                node_labels[i] = max(votes.items(), key=lambda x: x[1])[0]
        
        edge_index = []
        edge_attr = []
        
        for i, (src_ip, dst_ip) in enumerate(zip(src_ips, dst_ips)):
            src_idx = self.ip_to_idx[src_ip]
            dst_idx = self.ip_to_idx[dst_ip]
            
            edge_index.append([src_idx, dst_idx])
            edge_attr.append(features[i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(node_labels, dtype=torch.long)
        ).to(device)
        
        print(f"IP-based Graph: {graph_data.num_nodes:,} nodes, {graph_data.num_edges:,} edges, avg degree: {graph_data.num_edges / graph_data.num_nodes:.2f}")
        
        return graph_data


class FlowBasedKNNGraphBuilder:
    def __init__(self, k=10, metric='cosine'):
        self.k = k
        self.metric = metric
    
    def build(self, features, labels, device='cpu'):
        n_samples = len(features)
        k_actual = min(self.k, n_samples - 1)
        
        adj_matrix = kneighbors_graph(
            features,
            n_neighbors=k_actual,
            metric=self.metric,
            mode='connectivity',
            include_self=False,
            n_jobs=-1
        )
        
        edge_index = torch.tensor(
            np.array(adj_matrix.nonzero()),
            dtype=torch.long
        )
        
        graph_data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long)
        ).to(device)
        
        print(f"Flow-based KNN Graph (k={k_actual}, metric={self.metric}): {graph_data.num_nodes:,} nodes, {graph_data.num_edges:,} edges, avg degree: {graph_data.num_edges / graph_data.num_nodes:.2f}")
        
        return graph_data