import threading
import time
import argparse
import sys
from traffic_sender import TrafficSender

def run_detector(detector):
    """Thread chạy IDS Detector"""
    try:
        detector.start_server()
    except Exception as e:
        print(f"❌ Detector error: {e}")
        import traceback
        traceback.print_exc()

def run_sender(sender, delay=0.3):
    """Thread chạy Traffic Sender"""
    try:
        time.sleep(2)  # Đợi detector khởi động
        sender.send_flows(delay=delay)
    except Exception as e:
        print(f"❌ Sender error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoT IDS Real-time Demo (GNN vs CNN-LSTM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo CNN-LSTM (default)
  python demo_standalone.py
  
  # Demo GNN
  python demo_standalone.py --model gnn
  
  # Custom flows and delay
  python demo_standalone.py --model gnn --flows 200 --delay 0.2
  
  # Custom paths
  python demo_standalone.py --model cnn-lstm \\
      --cnn-model "./CNN-LSTM/Time-Based Split/cnn+lstm/hybrid_cnn+lstm_best.keras" \\
      --cnn-scaler "./CNN-LSTM/Time-Based Split/cnn+lstm/scaler.pkl"
        """
    )
    
    # ==================== MODEL SELECTION ====================
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["gnn", "cnn-lstm"], 
        default="cnn-lstm",
        help="Model to use: 'gnn' or 'cnn-lstm' (default: cnn-lstm)"
    )
    
    # ==================== COMMON PARAMETERS ====================
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host for socket communication (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=9999,
        help="Port for socket communication (default: 9999)"
    )
    parser.add_argument(
        "--flows", 
        type=int, 
        default=100,
        help="Number of flows to demo (default: 100)"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.3,
        help="Delay between flows in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--data-folder", 
        type=str, 
        default="./cleaned_data",
        help="Folder containing test data (default: ./cleaned_data)"
    )
    
    # ==================== GNN-SPECIFIC PARAMETERS ====================
    gnn_group = parser.add_argument_group('GNN Options')
    gnn_group.add_argument(
        "--gnn-checkpoint", 
        type=str,
        default="./saved_gnn/gnn_checkpoint.pth",
        help="Path to GNN checkpoint file (default: ./saved_gnn/gnn_checkpoint.pth)"
    )
    gnn_group.add_argument(
        "--gnn-scaler", 
        type=str,
        default="./saved_gnn/gnn_scaler.pkl",
        help="Path to GNN scaler file (default: ./saved_gnn/gnn_scaler.pkl)"
    )
    gnn_group.add_argument(
        "--k-neighbors", 
        type=int, 
        default=10,
        help="K neighbors for GNN graph construction (default: 10)"
    )
    
    # ==================== CNN-LSTM SPECIFIC PARAMETERS ====================
    cnn_group = parser.add_argument_group('CNN-LSTM Options')
    cnn_group.add_argument(
        "--cnn-model", 
        type=str,
        default="./CNN-LSTM/Time-Based Split/cnn_lstm/hybrid_cnn_lstm_best.keras",
        help="Path to CNN-LSTM model file"
    )
    cnn_group.add_argument(
        "--cnn-scaler", 
        type=str,
        default="./CNN-LSTM/Time-Based Split/cnn_lstm/scaler.pkl",
        help="Path to CNN-LSTM scaler file"
    )
    
    args = parser.parse_args()
    
    # ==================== PRINT BANNER ====================
    print("\n" + "=" * 70)
    print("🔥 IoT IDS Real-time Demo".center(70))
    print("=" * 70)
    print(f"{'Model:':<20} {args.model.upper()}")
    print(f"{'Connection:':<20} {args.host}:{args.port}")
    print(f"{'Test Flows:':<20} {args.flows}")
    print(f"{'Flow Delay:':<20} {args.delay}s")
    print(f"{'Data Folder:':<20} {args.data_folder}")
    print("=" * 70 + "\n")
    
    try:
        # ==================== INITIALIZE DETECTOR ====================
        if args.model == "gnn":
            print("🧠 Initializing GNN Detector...\n")
            from ids_detector_gnn import IDSDetectorGNN
            
            detector = IDSDetectorGNN(
                checkpoint_path=args.gnn_checkpoint,
                scaler_path=args.gnn_scaler,
                host=args.host,
                port=args.port,
            )
        
        elif args.model == "cnn-lstm":
            print("Initializing CNN-LSTM Detector...\n")
            from ids_detector import IDSDetector
            
            detector = IDSDetector(
                model_path=args.cnn_model,
                scaler_path=args.cnn_scaler,
                host=args.host,
                port=args.port
            )
        
        # ==================== INITIALIZE SENDER ====================
        print(f"\n📡 Initializing Traffic Sender...")
        sender = TrafficSender(
            data_folder=args.data_folder,
            host=args.host,
            port=args.port
        )
        
        sender.load_data(max_flows=args.flows)
        
        # ==================== CREATE THREADS ====================
        detector_thread = threading.Thread(
            target=run_detector, 
            args=(detector,), 
            daemon=True,
            name="DetectorThread"
        )
        sender_thread = threading.Thread(
            target=run_sender, 
            args=(sender, args.delay),
            name="SenderThread"
        )
        
        # ==================== START DEMO ====================
        print("\n" + "=" * 70)
        print("▶️  STARTING DEMO...".center(70))
        print("=" * 70 + "\n")
        
        detector_thread.start()
        sender_thread.start()
        
        # ==================== WAIT FOR COMPLETION ====================
        sender_thread.join()
        time.sleep(1)  # Đợi detector xử lý hết
        
        print("\n" + "=" * 70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!".center(70))
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("⚠️  INTERRUPTED BY USER".center(70))
        print("=" * 70)
        sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("\n💡 Tips:")
        print("   - Make sure model files exist")
        print("   - Run 'python save_scaler.py' for CNN-LSTM")
        print("   - Run 'python save_scaler_gnn.py' for GNN")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
