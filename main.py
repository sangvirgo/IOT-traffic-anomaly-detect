import torch
import json
import argparse
from pathlib import Path

from load_gnn_data import CICIDS_Loader
from graph_builder import IPBasedGraphBuilder, FlowBasedKNNGraphBuilder
from activelearn import ActiveLearningEngine
from evaluation import FullySupervisedBaseline, visualize_comparison, print_summary_table

class Config:
    RAW_DATA_DIR = Path('./raw_data')
    RESULTS_DIR = Path('./results')
    
    FILES_NO_IP = [
        '02-14-2018.csv',
        '02-15-2018.csv',
        '02-16-2018.csv'
    ]
    FILE_WITH_IP = '02-20-2018.csv'
    
    SAMPLE_PER_FILE = 50000
    K_NEIGHBORS = 10
    INITIAL_BUDGET = 0.05
    QUERY_BUDGET = 0.02
    N_ROUNDS = 5
    AL_STRATEGY = 'entropy'
    EPOCHS_AL = 50
    EPOCHS_FULL = 100
    LEARNING_RATE = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_flow_based_knn(config):
    print(f"\n{'='*70}\nAPPROACH 1: FLOW-BASED KNN GNN\n{'='*70}")
    
    results = {}
    
    try:
        loader = CICIDS_Loader(config.RAW_DATA_DIR)
        df = loader.load_selected_files(config.FILES_NO_IP, sample_per_file=config.SAMPLE_PER_FILE)
        
        X, y, features, _, _ = loader.preprocess(df)
        
        knn_builder = FlowBasedKNNGraphBuilder(k=config.K_NEIGHBORS)
        graph_knn = knn_builder.build(X, y, device=config.DEVICE)
        
        print("\nRunning Active Learning...")
        al_engine = ActiveLearningEngine(
            graph_data=graph_knn,
            true_labels=y,
            initial_budget=config.INITIAL_BUDGET,
            query_budget=config.QUERY_BUDGET,
            n_rounds=config.N_ROUNDS,
            approach_name='Flow-based KNN'
        )
        
        history_al, model_al = al_engine.run(al_strategy=config.AL_STRATEGY, verbose=True)
        results['al'] = history_al
        
        print("\nRunning Fully Supervised Baseline...")
        baseline = FullySupervisedBaseline(
            graph_data=graph_knn,
            labels=y,
            approach_name='Flow-based KNN'
        )
        
        results_full, model_full = baseline.run(test_size=0.2, epochs=config.EPOCHS_FULL, verbose=True)
        results['full'] = results_full
        
        torch.save(model_al.state_dict(), config.RESULTS_DIR / 'model_flow_knn_al.pth')
        torch.save(model_full.state_dict(), config.RESULTS_DIR / 'model_flow_knn_full.pth')
        
        print("\nFlow-based KNN approach completed!")
        
    except Exception as e:
        print(f"\nError in Flow-based KNN: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def run_ip_based(config):
    print(f"\n{'='*70}\nAPPROACH 2: IP-BASED GNN\n{'='*70}")
    
    results = {}
    
    try:
        loader = CICIDS_Loader(config.RAW_DATA_DIR)
        df = loader.load_selected_files([config.FILE_WITH_IP], sample_per_file=config.SAMPLE_PER_FILE)
        
        X, y, features, src_ips, dst_ips = loader.preprocess(df)
        
        if src_ips is None:
            print("Warning: No IP addresses found! Skipping IP-based approach.")
            return results
        
        ip_builder = IPBasedGraphBuilder()
        graph_ip = ip_builder.build(X, y, src_ips, dst_ips, device=config.DEVICE)
        
        print("\nRunning Active Learning...")
        al_engine = ActiveLearningEngine(
            graph_data=graph_ip,
            true_labels=y,
            initial_budget=config.INITIAL_BUDGET,
            query_budget=config.QUERY_BUDGET,
            n_rounds=config.N_ROUNDS,
            approach_name='IP-based'
        )
        
        history_al, model_al = al_engine.run(al_strategy=config.AL_STRATEGY, verbose=True)
        results['al'] = history_al
        
        print("\nRunning Fully Supervised Baseline...")
        baseline = FullySupervisedBaseline(
            graph_data=graph_ip,
            labels=y,
            approach_name='IP-based'
        )
        
        results_full, model_full = baseline.run(test_size=0.2, epochs=config.EPOCHS_FULL, verbose=True)
        results['full'] = results_full
        
        torch.save(model_al.state_dict(), config.RESULTS_DIR / 'model_ip_based_al.pth')
        torch.save(model_full.state_dict(), config.RESULTS_DIR / 'model_ip_based_full.pth')
        
        print("\nIP-based approach completed!")
        
    except Exception as e:
        print(f"\nError in IP-based: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main(config):
    print(f"\n{'='*70}\nDUAL GNN APPROACH: IP-based vs Flow-based KNN\nActive Learning vs Fully Supervised\n{'='*70}")
    print(f"\nDevice: {config.DEVICE}")
    print(f"Sample per file: {config.SAMPLE_PER_FILE if config.SAMPLE_PER_FILE else 'ALL'}")
    print(f"K-neighbors: {config.K_NEIGHBORS}")
    print(f"AL Strategy: {config.AL_STRATEGY}")
    
    config.RESULTS_DIR.mkdir(exist_ok=True)
    
    all_results = {}
    
    flow_results = run_flow_based_knn(config)
    if flow_results:
        all_results['flow_knn_al'] = flow_results['al']
        all_results['flow_knn_full'] = flow_results['full']
    
    ip_results = run_ip_based(config)
    if ip_results:
        all_results['ip_based_al'] = ip_results['al']
        all_results['ip_based_full'] = ip_results['full']
    
    if all_results:
        print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}")
        
        with open(config.RESULTS_DIR / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved: {config.RESULTS_DIR / 'all_results.json'}")
        
        print(f"\n{'='*70}\nGENERATING VISUALIZATIONS\n{'='*70}")
        visualize_comparison(all_results, save_dir=config.RESULTS_DIR)
        print_summary_table(all_results)
    
    print(f"\n{'='*70}\nEXPERIMENT COMPLETED!\n{'='*70}")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"  - all_results.json")
    print(f"  - model_*.pth")
    print(f"  - al_comparison.png")
    print(f"  - final_comparison.png")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual GNN + Active Learning for CICIDS 2018')
    
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full', 'custom'],
                       help='Execution mode')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size per file (None = load all)')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of K neighbors for KNN graph')
    parser.add_argument('--strategy', type=str, default='entropy', choices=['entropy', 'margin', 'random'],
                       help='Active Learning query strategy')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == 'quick':
        config.SAMPLE_PER_FILE = 50000
        print("Running in QUICK mode (50K samples per file)")
    elif args.mode == 'full':
        config.SAMPLE_PER_FILE = None
        print("Running in FULL mode (all data)")
    
    if args.sample is not None:
        config.SAMPLE_PER_FILE = args.sample
    
    config.K_NEIGHBORS = args.k
    config.AL_STRATEGY = args.strategy
    
    if args.device != 'auto':
        config.DEVICE = args.device
    
    results = main(config)
    
    print("\nDone!")