"""
Standalone Demo - Chạy cả Sender và Detector trên 1 máy
"""
import threading
import time
import sys
import traceback

# Import từ 2 file trên
from traffic_sender import TrafficSender
from ids_detector import IDSDetector

def run_detector(detector):
    """Thread chạy IDS Detector"""
    try:
        detector.start_server()
    except Exception as e:
        print(f"\n❌ Detector error: {e}")
        traceback.print_exc()

def run_sender(sender, delay=0.5):
    """Thread chạy Traffic Sender"""
    try:
        time.sleep(2)  # Đợi detector khởi động
        sender.send_flows(delay=delay)
    except Exception as e:
        print(f"\n❌ Sender error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("     🚨 IoT IDS Real-time Demo (Standalone Mode)")
    print("="*60)
    print("\n[+] Starting IDS Detector and Traffic Sender on same machine...\n")
    
    try:
        # Khởi tạo
        detector = IDSDetector(
            model_path='./CNN-LSTM/Time-Based Split/cnn_lstm/hybrid_cnn_lstm_best.keras',
            scaler_path='./CNN-LSTM/Time-Based Split/cnn_lstm/scaler.pkl',
            host='127.0.0.1',
            port=9999
        )
        
        sender = TrafficSender(
            data_folder='./cleaned_data',
            host='127.0.0.1',
            port=9999
        )
        
        sender.load_data(max_flows=100)  # Demo với 100 flows
        
        # Tạo threads
        detector_thread = threading.Thread(target=run_detector, args=(detector,), daemon=True)
        sender_thread = threading.Thread(target=run_sender, args=(sender, 0.3))
        
        # Start
        detector_thread.start()
        sender_thread.start()
        
        # Wait
        sender_thread.join()
        
        # Give detector time to finish processing
        time.sleep(1)
        
        print("\n[+] Demo finished!")
        
    except KeyboardInterrupt:
        print("\n\n[+] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
