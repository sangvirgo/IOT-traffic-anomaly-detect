"""
IDS Detector Server - Nhận flow và detect bằng CNN-LSTM
"""
import socket
import json
import pickle
import numpy as np
from tensorflow import keras
from collections import deque
import time
import traceback

class IDSDetector:
    def __init__(self, model_path='./CNN-LSTM/Time-Based Split/cnn_lstm/hybrid_cnn_lstm_best.keras',
                 scaler_path='./CNN-LSTM/Time-Based Split/cnn_lstm/scaler.pkl',
                 host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.sequence_length = 10
        self.n_features = 77
        
        # Load model và scaler
        print("[+] Loading model and scaler...")
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # VERIFY scaler type
        if not hasattr(self.scaler, 'transform'):
            raise TypeError(f"❌ scaler.pkl must be StandardScaler, got {type(self.scaler)}\n"
                          "    Run 'python save_scaler.py' to create correct scaler.pkl")
        
        print(f"[+] Model loaded successfully!")
        print(f"[+] Scaler type: {type(self.scaler).__name__}")
        
        # Buffer để tạo sequence (10 timesteps)
        self.flow_buffer = deque(maxlen=self.sequence_length)
        
        # Statistics
        self.stats = {
            'total': 0,
            'detected_attack': 0,
            'detected_benign': 0,
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0
        }
    
    def preprocess_flow(self, features):
        """Chuẩn hóa flow và tạo sequence"""
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Ensure correct shape (77,)
        if features.shape != (self.n_features,):
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
        
        # Normalize
        features_normalized = self.scaler.transform([features])[0]
        
        # Thêm vào buffer
        self.flow_buffer.append(features_normalized)
        
        # Nếu chưa đủ 10 flows, pad bằng zeros
        if len(self.flow_buffer) < self.sequence_length:
            padded = np.zeros((self.sequence_length, self.n_features))
            for i, flow in enumerate(self.flow_buffer):
                padded[i] = flow
            sequence = padded
        else:
            sequence = np.array(self.flow_buffer)
        
        # Reshape cho model: (1, 10, 77)
        return sequence.reshape(1, self.sequence_length, self.n_features)
    
    def predict(self, features):
        """Predict flow"""
        sequence = self.preprocess_flow(features)
        
        # Inference
        start_time = time.time()
        prob = self.model.predict(sequence, verbose=0)[0][0]
        latency = (time.time() - start_time) * 1000  # ms
        
        prediction = 1 if prob >= 0.5 else 0
        confidence = abs(prob - 0.5) * 2  # Scale 0-1
        
        return {
            'prediction': prediction,
            'probability': float(prob),
            'confidence': float(confidence),
            'latency_ms': round(latency, 2)
        }
    
    def update_stats(self, prediction, true_label):
        """Update confusion matrix"""
        self.stats['total'] += 1
        
        if prediction == 1:
            self.stats['detected_attack'] += 1
        else:
            self.stats['detected_benign'] += 1
        
        if prediction == 1 and true_label == 1:
            self.stats['true_positive'] += 1
        elif prediction == 1 and true_label == 0:
            self.stats['false_positive'] += 1
        elif prediction == 0 and true_label == 0:
            self.stats['true_negative'] += 1
        elif prediction == 0 and true_label == 1:
            self.stats['false_negative'] += 1
    
    def print_stats(self):
        """In thống kê"""
        tp = self.stats['true_positive']
        fp = self.stats['false_positive']
        tn = self.stats['true_negative']
        fn = self.stats['false_negative']
        
        accuracy = (tp + tn) / self.stats['total'] * 100 if self.stats['total'] > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*60)
        print("📊 DETECTION STATISTICS")
        print("="*60)
        print(f"Total Flows:       {self.stats['total']}")
        print(f"Detected Attack:   {self.stats['detected_attack']} (🔴)")
        print(f"Detected Benign:   {self.stats['detected_benign']} (🟢)")
        print(f"\nConfusion Matrix:")
        print(f"  True Positive:   {tp}")
        print(f"  False Positive:  {fp}")
        print(f"  True Negative:   {tn}")
        print(f"  False Negative:  {fn}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.2f}%")
        print(f"  Precision: {precision:.2f}%")
        print(f"  Recall:    {recall:.2f}%")
        print(f"  F1-Score:  {f1:.2f}%")
        print("="*60 + "\n")
    
    def start_server(self):
        """Khởi động IDS server"""
        print(f"[+] Starting IDS Detector on {self.host}:{self.port}...")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            
            print(f"[+] IDS Detector listening on port {self.port}")
            print("[+] Waiting for Traffic Sender...\n")
            
            conn, addr = server_sock.accept()
            print(f"[+] Connected by {addr}\n")
            
            buffer = ""
            
            with conn:
                try:
                    while True:
                        data = conn.recv(4096).decode('utf-8')
                        if not data:
                            break
                        
                        buffer += data
                        
                        # Xử lý từng packet (mỗi packet 1 dòng JSON)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            
                            try:
                                packet = json.loads(line)
                                flow_id = packet['flow_id']
                                features = packet['features']
                                true_label = packet['true_label']
                                
                                # Predict
                                result = self.predict(features)
                                prediction = result['prediction']
                                
                                # Update stats
                                self.update_stats(prediction, true_label)
                                
                                # Display
                                pred_str = "🔴 ATTACK" if prediction == 1 else "🟢 BENIGN"
                                true_str = "🔴 ATTACK" if true_label == 1 else "🟢 BENIGN"
                                correct = "✓" if prediction == true_label else "✗"
                                
                                print(f"Flow {flow_id} | Pred: {pred_str} | True: {true_str} | "
                                      f"Prob: {result['probability']:.3f} | "
                                      f"Latency: {result['latency_ms']:.2f}ms | {correct}")
                                
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                print(f"⚠️  Error processing flow: {e}")
                                traceback.print_exc()
                                continue
                
                except Exception as e:
                    print(f"\n❌ Server error: {e}")
                    traceback.print_exc()
                finally:
                    print("\n[+] Connection closed")
                    self.print_stats()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='IDS Detector Server')
    parser.add_argument('--model', default='./CNN-LSTM/Time-Based Split/cnn_lstm/hybrid_cnn_lstm_best.keras', 
                        help='Path to model file')
    parser.add_argument('--scaler', default='./CNN-LSTM/Time-Based Split/cnn_lstm/scaler.pkl',
                        help='Path to scaler file')
    parser.add_argument('--host', default='0.0.0.0', help='Listen host')
    parser.add_argument('--port', type=int, default=9999, help='Listen port')
    
    args = parser.parse_args()
    
    try:
        detector = IDSDetector(
            model_path=args.model,
            scaler_path=args.scaler,
            host=args.host,
            port=args.port
        )
        
        detector.start_server()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
