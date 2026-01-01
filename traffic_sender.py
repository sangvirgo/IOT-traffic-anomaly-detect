"""
Traffic Sender - Đóng giả attacker gửi flow mạng
Đọc từ nhiều file CSV cleaned và gửi qua socket
"""
import socket
import json
import time
import pandas as pd
import glob
import random
from pathlib import Path

class TrafficSender:
    def __init__(self, data_folder='./cleaned_data', host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.data_folder = data_folder
        self.flows = []
        
    def load_data(self, max_flows=1000):
        """Load flows từ nhiều file CSV"""
        print(f"[+] Loading data from {self.data_folder}...")
        csv_files = glob.glob(f"{self.data_folder}/*.csv")
        
        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file CSV trong {self.data_folder}")
        
        print(f"[+] Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Lấy 77 features + Label
                if 'Label' in df.columns:
                    self.flows.append(df)
                    print(f"    ✓ {Path(csv_file).name}: {len(df)} flows")
            except Exception as e:
                print(f"    ✗ Error reading {csv_file}: {e}")
        
        # Merge tất cả flows
        if self.flows:
            self.flows = pd.concat(self.flows, ignore_index=True)
            
            # Sample nếu quá nhiều
            if len(self.flows) > max_flows:
                self.flows = self.flows.sample(n=max_flows, random_state=42)
            
            print(f"[+] Total flows loaded: {len(self.flows)}")
            print(f"    - Benign: {len(self.flows[self.flows['Label']==0])}")
            print(f"    - Attack: {len(self.flows[self.flows['Label']==1])}")
        else:
            raise ValueError("Không load được flows nào!")
    
    def send_flows(self, delay=0.1, shuffle=True):
        """Gửi flows qua socket"""
        if shuffle:
            self.flows = self.flows.sample(frac=1).reset_index(drop=True)
        
        print(f"\n[+] Connecting to IDS Detector at {self.host}:{self.port}...")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))
            print("[+] Connected! Starting to send flows...\n")
            
            for idx, row in self.flows.iterrows():
                # Chuẩn bị packet
                features = row.drop('Label').values.tolist()  # 77 features
                true_label = int(row['Label'])
                
                packet = {
                    'flow_id': idx,
                    'features': features,
                    'true_label': true_label  # Ground truth (để kiểm tra)
                }
                
                # Gửi
                message = json.dumps(packet) + '\n'
                sock.sendall(message.encode('utf-8'))
                
                # Log
                label_str = "🔴 ATTACK" if true_label == 1 else "🟢 BENIGN"
                print(f"[{idx+1}/{len(self.flows)}] Sent flow {idx} | {label_str}")
                
                time.sleep(delay)
            
            print("\n[+] All flows sent!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Sender (Attacker Simulator)')
    parser.add_argument('--data', default='./data', help='Folder chứa các file CSV cleaned')
    parser.add_argument('--host', default='127.0.0.1', help='IDS Detector IP')
    parser.add_argument('--port', type=int, default=9999, help='IDS Detector Port')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay giữa các flow (giây)')
    parser.add_argument('--max-flows', type=int, default=500, help='Số flow tối đa')
    
    args = parser.parse_args()
    
    sender = TrafficSender(data_folder=args.data, host=args.host, port=args.port)
    sender.load_data(max_flows=args.max_flows)
    sender.send_flows(delay=args.delay)
