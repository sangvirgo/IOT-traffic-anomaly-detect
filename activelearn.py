import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import time

from models import GCN_IDS

class ActiveLearningEngine:
    def __init__(self, graph_data, true_labels, 
                 initial_budget=0.05, query_budget=0.02, n_rounds=5,
                 approach_name='Unknown'):
        self.graph_data = graph_data
        self.true_labels = true_labels.copy()
        self.n_samples = len(true_labels)
        self.device = graph_data.x.device
        self.approach_name = approach_name
        
        self.initial_budget = int(self.n_samples * initial_budget)
        self.query_budget = int(self.n_samples * query_budget)
        self.n_rounds = n_rounds
        
        self.train_mask = torch.zeros(self.n_samples, dtype=torch.bool)
        self.test_mask = torch.ones(self.n_samples, dtype=torch.bool)
        
        self.history = {
            'approach': approach_name,
            'round': [],
            'labeled_count': [],
            'labeled_percent': [],
            'test_acc': [],
            'test_f1': [],
            'test_auc': [],
            'train_time': []
        }
    
    def initial_labeling(self, strategy='stratified'):
        if strategy == 'stratified':
            benign_idx = np.where(self.true_labels == 0)[0]
            attack_idx = np.where(self.true_labels == 1)[0]
            
            benign_ratio = len(benign_idx) / self.n_samples
            attack_ratio = len(attack_idx) / self.n_samples
            
            n_benign = int(self.initial_budget * benign_ratio)
            n_attack = self.initial_budget - n_benign
            
            selected_benign = np.random.choice(benign_idx, n_benign, replace=False)
            selected_attack = np.random.choice(attack_idx, n_attack, replace=False)
            
            initial_indices = np.concatenate([selected_benign, selected_attack])
        else:
            initial_indices = np.random.choice(self.n_samples, self.initial_budget, replace=False)
        
        self.train_mask[initial_indices] = True
        self.test_mask[initial_indices] = False
        
        print(f"Initial labeled: {len(initial_indices):,}/{self.n_samples:,} ({len(initial_indices)/self.n_samples*100:.1f}%)")
    
    def train_model(self, epochs=50, lr=0.01, verbose=True):
        model = GCN_IDS(input_dim=self.graph_data.x.shape[1], hidden_dim=64).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        self.graph_data.train_mask = self.train_mask.to(self.device)
        
        model.train()
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            optimizer.zero_grad()
            out = model(self.graph_data)
            loss = criterion(out[self.train_mask], self.graph_data.y[self.train_mask])
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                pred = out[self.train_mask].argmax(dim=1)
                train_acc = (pred == self.graph_data.y[self.train_mask]).float().mean()
                pbar.set_postfix({'loss': f'{loss:.4f}', 'train_acc': f'{train_acc:.4f}'})
        
        return model
    
    def evaluate(self, model):
        model.eval()
        
        with torch.no_grad():
            out = model(self.graph_data)
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)
            
            test_pred = pred[self.test_mask].cpu().numpy()
            test_true = self.graph_data.y[self.test_mask].cpu().numpy()
            test_probs = probs[self.test_mask, 1].cpu().numpy()
            
            acc = accuracy_score(test_true, test_pred)
            f1 = f1_score(test_true, test_pred, average='binary')
            
            if len(np.unique(test_true)) > 1:
                auc = roc_auc_score(test_true, test_probs)
            else:
                auc = 0.0
        
        return acc, f1, auc
    
    def select_uncertain(self, model, strategy='entropy'):
        model.eval()
        
        with torch.no_grad():
            out = model(self.graph_data)
            probs = F.softmax(out, dim=1)
            
            if strategy == 'entropy':
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                entropy[self.train_mask] = -1
                _, uncertain_idx = torch.topk(entropy, self.query_budget)
            
            elif strategy == 'margin':
                top2, _ = torch.topk(probs, k=2, dim=1)
                margin = top2[:, 0] - top2[:, 1]
                margin[self.train_mask] = float('inf')
                _, uncertain_idx = torch.topk(-margin, self.query_budget)
            
            elif strategy == 'random':
                unlabeled_indices = torch.where(~self.train_mask)[0]
                perm = torch.randperm(len(unlabeled_indices))
                uncertain_idx = unlabeled_indices[perm[:self.query_budget]]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        return uncertain_idx.cpu().numpy()
    
    def run(self, al_strategy='entropy', verbose=True):
        print(f"\nActive Learning: {self.approach_name} | Strategy: {al_strategy} | Dataset: {self.n_samples:,} samples")
        
        self.initial_labeling(strategy='stratified')
        
        for round_idx in range(self.n_rounds):
            print(f"\nRound {round_idx + 1}/{self.n_rounds} - Labeled: {self.train_mask.sum():,}/{self.n_samples:,} ({self.train_mask.sum()/self.n_samples*100:.1f}%)")
            
            start_time = time.time()
            model = self.train_model(epochs=50, verbose=verbose)
            train_time = time.time() - start_time
            
            test_acc, test_f1, test_auc = self.evaluate(model)
            
            print(f"Results: Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}, Time={train_time:.1f}s")
            
            self.history['round'].append(round_idx + 1)
            self.history['labeled_count'].append(self.train_mask.sum().item())
            self.history['labeled_percent'].append(self.train_mask.sum().item() / self.n_samples * 100)
            self.history['test_acc'].append(test_acc)
            self.history['test_f1'].append(test_f1)
            self.history['test_auc'].append(test_auc)
            self.history['train_time'].append(train_time)
            
            if round_idx < self.n_rounds - 1:
                uncertain_idx = self.select_uncertain(model, strategy=al_strategy)
                self.train_mask[uncertain_idx] = True
                self.test_mask[uncertain_idx] = False
        
        print(f"\nFinal - {self.approach_name}: Labeled {self.train_mask.sum():,}/{self.n_samples:,} ({self.train_mask.sum()/self.n_samples*100:.1f}%), Acc={self.history['test_acc'][-1]:.4f}, F1={self.history['test_f1'][-1]:.4f}, AUC={self.history['test_auc'][-1]:.4f}")
        
        return self.history, model