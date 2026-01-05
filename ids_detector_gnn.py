# ids_detector_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import numpy as np
import pickle, socket, json

class GCN_IDS(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class IDSDetectorGNN:
    def __init__(self,
                 checkpoint_path="./saved_models/gnn_checkpoint.pth",
                 scaler_path="./saved_models/gnn_scaler.pkl",
                 host="0.0.0.0", port=9999):
        self.host = host
        self.port = port
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["model_config"]
        self.n_features = cfg["input_dim"]

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model = GCN_IDS(cfg["input_dim"], cfg["hidden_dim"],
                            cfg["num_layers"], cfg.get("dropout", 0.3)).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.flow_buffer = []
        self.stats = {
            "total": 0, "detected_attack": 0, "detected_benign": 0,
            "true_positive": 0, "false_positive": 0,
            "true_negative": 0, "false_negative": 0,
        }

    def _build_graph(self, feats: np.ndarray) -> Data:
        n = feats.shape[0]
        k = min(10, n - 1)
        adj = kneighbors_graph(feats, n_neighbors=k, metric="cosine",
                               mode="connectivity", include_self=False)
        edge_index = np.array(adj.nonzero())
        return Data(
            x=torch.tensor(feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long)
        ).to(self.device)

    def _predict_one(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (self.n_features,):
            raise ValueError(f"Expected {self.n_features} features, got {x.shape}")
        x_norm = self.scaler.transform(x.reshape(1, -1))[0]
        self.flow_buffer.append(x_norm)

        if len(self.flow_buffer) < 10:
            buf = np.array(self.flow_buffer)
            while buf.shape[0] < 10:
                buf = np.vstack([buf, buf[-1]])
        else:
            buf = np.array(self.flow_buffer[-50:])

        graph = self._build_graph(buf)
        with torch.no_grad():
            probs = torch.exp(self.model(graph))[-1].cpu().numpy()
        pred = int(probs[1] > probs[0])
        return pred, float(probs[1])

    def _update_stats(self, pred, true):
        s = self.stats
        s["total"] += 1
        if pred == 1:
            s["detected_attack"] += 1
        else:
            s["detected_benign"] += 1
        if pred == 1 and true == 1:
            s["true_positive"] += 1
        elif pred == 1 and true == 0:
            s["false_positive"] += 1
        elif pred == 0 and true == 0:
            s["true_negative"] += 1
        else:
            s["false_negative"] += 1

    def _print_stats(self):
        s = self.stats
        tp, fp, tn, fn = s["true_positive"], s["false_positive"], s["true_negative"], s["false_negative"]
        total = s["total"] or 1
        acc = (tp + tn) / total * 100
        prec = tp / (tp + fp) * 100 if tp + fp > 0 else 0
        rec = tp / (tp + fn) * 100 if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        print("Total:", s["total"])
        print("Detected Attack:", s["detected_attack"], "| Detected Benign:", s["detected_benign"])
        print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
        print(f"Accuracy: {acc:.2f}%  Precision: {prec:.2f}%  Recall: {rec:.2f}%  F1: {f1:.2f}%")

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"GNN detector listening on {self.host}:{self.port}")
            conn, addr = srv.accept()
            print("Connected by", addr)
            buf = ""
            with conn:
                while True:
                    data = conn.recv(4096).decode("utf-8")
                    if not data:
                        break
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        try:
                            pkt = json.loads(line)
                            flow_id = pkt["flow_id"]
                            feats = pkt["features"]
                            true_label = int(pkt["true_label"])
                            pred, p_attack = self._predict_one(feats)
                            self._update_stats(pred, true_label)
                            pred_str = "ATTACK" if pred == 1 else "BENIGN"
                            true_str = "ATTACK" if true_label == 1 else "BENIGN"
                            ok = "✓" if pred == true_label else "✗"
                            print(f"Flow {flow_id}: Pred={pred_str} True={true_str} Prob={p_attack:.3f} {ok}")
                        except Exception:
                            continue
            print("Connection closed")
            self._print_stats()
