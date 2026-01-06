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
                 checkpoint_path="./saved_gnn/gnn_checkpoint.pth",
                 scaler_path="./saved_gnn/gnn_scaler.pkl",
                 host="0.0.0.0", port=9999,
                 buffer_size=50):  # ← Thêm param
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["model_config"]
        self.n_features = cfg["input_dim"]
        
        print(f"Model expects {self.n_features} features")

        # Load scaler
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        # === CRITICAL: Verify feature count ===
        if hasattr(self.scaler, 'n_features_in_'):
            scaler_features = self.scaler.n_features_in_
            if scaler_features != self.n_features:
                raise ValueError(
                    f"Scaler expects {scaler_features} features, "
                    f"but model needs {self.n_features}! "
                    f"Run save_scaler_gnn.py again."
                )
        
        print(f"✓ Scaler has {self.scaler.mean_.shape[0]} features (matches model)")

        # Load model
        print("Loading model...")
        self.model = GCN_IDS(
            cfg["input_dim"],
            cfg["hidden_dim"],
            cfg["num_layers"],
            cfg.get("dropout", 0.3)
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print("✓ Model loaded successfully\n")

        # Buffer + stats
        self.flow_buffer = []
        self.stats = {
            "total": 0, "detected_attack": 0, "detected_benign": 0,
            "true_positive": 0, "false_positive": 0,
            "true_negative": 0, "false_negative": 0,
        }

        # threshold
        self.attack_threshold = 0.35  

    def _build_graph(self, feats: np.ndarray) -> Data:
        """Build KNN graph from features"""
        n = feats.shape[0]
        k = min(10, n - 1)
        
        if n < 3:
            # Tạo self-loop cho graph nhỏ
            edge_index = torch.tensor([[i, i] for i in range(n)], dtype=torch.long).t()
        else:
            adj = kneighbors_graph(
                feats,
                n_neighbors=k,
                metric="cosine",
                mode="connectivity",
                include_self=False,
            )
            edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
        
        return Data(
            x=torch.tensor(feats, dtype=torch.float32),
            edge_index=edge_index
        ).to(self.device)

    def _predict_one(self, features):
        """Predict single flow"""
        x = np.asarray(features, dtype=np.float32)
        
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Feature mismatch! Model expects {self.n_features} features, "
                f"got {x.shape[0]}. Check preprocessing in traffic_sender.py"
            )

        # Normalize
        x_norm = self.scaler.transform(x.reshape(1, -1))[0]
        self.flow_buffer.append(x_norm)
        
        # === FIX: Maintain buffer size ===
        if len(self.flow_buffer) > self.buffer_size:
            self.flow_buffer = self.flow_buffer[-self.buffer_size:]

        # === FIX: Better graph construction ===
        if len(self.flow_buffer) < 10:
            # Lấy toàn bộ buffer + pad bằng mean
            buf = np.array(self.flow_buffer)
            if buf.shape[0] < 10:
                # Pad bằng mean của buffer thay vì duplicate
                mean_vec = buf.mean(axis=0)
                pad_rows = np.tile(mean_vec, (10 - buf.shape[0], 1))
                buf = np.vstack([buf, pad_rows])
        else:
            buf = np.array(self.flow_buffer[-self.buffer_size:])

        # Build graph
        graph = self._build_graph(buf)

        # Inference
        with torch.no_grad():
            logits = self.model(graph)
            probs = torch.exp(logits)
            # Lấy node cuối cùng (flow hiện tại)
            p_attack = float(probs[-1, 1].cpu())

        # Predict
        pred = int(p_attack >= self.attack_threshold)
        return pred, p_attack

    def _update_stats(self, pred, true):
        """Update statistics"""
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
        """Print final statistics"""
        s = self.stats
        tp, fp, tn, fn = (
            s["true_positive"],
            s["false_positive"],
            s["true_negative"],
            s["false_negative"],
        )
        
        total = s["total"] or 1
        acc = (tp + tn) / total * 100
        prec = tp / (tp + fp) * 100 if tp + fp > 0 else 0
        rec = tp / (tp + fn) * 100 if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Total Flows:        {s['total']}")
        print(f"Detected Attack:    {s['detected_attack']:5d} ({s['detected_attack']/total*100:5.1f}%)")
        print(f"Detected Benign:    {s['detected_benign']:5d} ({s['detected_benign']/total*100:5.1f}%)")
        print()
        print(f"True Positives:     {tp:5d}")
        print(f"False Positives:    {fp:5d}")
        print(f"True Negatives:     {tn:5d}")
        print(f"False Negatives:    {fn:5d}")
        print()
        print(f"Accuracy:           {acc:6.2f}%")
        print(f"Precision:          {prec:6.2f}%")
        print(f"Recall:             {rec:6.2f}%")
        print(f"F1-Score:           {f1:6.2f}%")
        print("=" * 70)

    def start_server(self):
        """Start TCP server and process flows"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            
            print(f"🧠 GNN Detector listening on {self.host}:{self.port}")
            print(f"   Threshold: {self.attack_threshold}")
            print(f"   Buffer size: {self.buffer_size}")
            print(f"   Features: {self.n_features}")
            print(f"   Device: {self.device}\n")
            
            conn, addr = srv.accept()
            print(f"✓ Connected by {addr}\n")

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
                            
                            print(
                                f"Flow {flow_id:4d}: "
                                f"Pred={pred_str:6s} True={true_str:6s} "
                                f"Prob={p_attack:.3f} {ok}"
                            )
                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error processing flow: {e}")
                            continue

            print("\nConnection closed")
            self._print_stats()