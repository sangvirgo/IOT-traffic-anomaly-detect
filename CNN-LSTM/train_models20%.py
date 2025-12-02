"""
Train CNN and LSTM models for CICIDS2018 Network Intrusion Detection
Optimized for 16GB RAM systems

Usage:
    # Train CNN
    python train_models.py --model cnn --data_dir ./split_data --epochs 20 --batch_size 64

    # Train LSTM
    python train_models.py --model lstm --data_dir ./split_data --epochs 20 --batch_size 64
    
    # Train both
    python train_models.py --model both --data_dir ./split_data --epochs 20 --batch_size 64
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gc
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✓ Found {len(physical_devices)} GPU(s), memory growth enabled")
    except:
        pass


class NetworkIDSTrainer:
    def __init__(self, data_dir, model_type='cnn', output_dir='./results', use_subset=False):
        """
        Args:
            data_dir: Directory with split data
            model_type: 'cnn' or 'lstm'
            output_dir: Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_subset = use_subset
        
        # Create subdirectories
        self.model_dir = self.output_dir / 'models'
        self.plot_dir = self.output_dir / 'plots'
        self.log_dir = self.output_dir / 'logs'
        
        self.model_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.history = None
        self.metadata = None
        
    def load_metadata(self):
        """Load dataset metadata"""
        print("="*70)
        print("LOADING METADATA")
        print("="*70)
        
        with open(self.data_dir / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"\n✓ Dataset info:")
        print(f"  Features: {self.metadata['n_features']}")
        print(f"  Classes: {self.metadata['n_classes']}")
        print(f"  Train size: {self.metadata['train_size']:,}")
        print(f"  Val size: {self.metadata['val_size']:,}")
        print(f"  Test size: {self.metadata['test_size']:,}")
        
        return self.metadata
    
    def load_data_generator(self, X_path, y_path, batch_size=64, shuffle=True):
        """Memory-efficient data generator"""
        X = np.load(X_path, mmap_mode='r')  # Memory-mapped file
        y = np.load(y_path)
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Reshape for model
            if self.model_type == 'cnn':
                X_batch = X_batch.reshape(-1, X_batch.shape[1], 1)
            elif self.model_type == 'lstm':
                # Reshape to (batch, timesteps, features)
                # Use 11 timesteps (77 features = 7 * 11)
                timesteps = 11
                features_per_step = X_batch.shape[1] // timesteps
                X_batch = X_batch[:, :timesteps * features_per_step]
                X_batch = X_batch.reshape(-1, timesteps, features_per_step)
            
            # One-hot encode labels
            n_classes_actual = int(y_batch.max()) + 1
            n_classes_needed = max(n_classes_actual, self.metadata['n_classes'])
            y_batch = to_categorical(y_batch, num_classes=n_classes_needed)
            
            yield X_batch, y_batch
    
    def create_cnn_model(self, input_shape, num_classes):
        """Create 1D CNN model for network traffic classification"""
        print("\n" + "="*70)
        print("BUILDING CNN MODEL")
        print("="*70)
        
        model = models.Sequential([
            # First Conv Block
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape, name='conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv1D(128, 3, activation='relu', name='conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv1D(256, 3, activation='relu', name='conv3'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu', name='dense1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu', name='dense2'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        print(f"\n✓ CNN Model created")
        print(f"  Input shape: {input_shape}")
        print(f"  Output classes: {num_classes}")
        
        return model
    
    def create_lstm_model(self, input_shape, num_classes):
        """Create LSTM model for sequential traffic analysis"""
        print("\n" + "="*70)
        print("BUILDING LSTM MODEL")
        print("="*70)
        
        model = models.Sequential([
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True, name='lstm2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third LSTM layer
            layers.LSTM(32, return_sequences=False, name='lstm3'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu', name='dense1'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu', name='dense2'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        print(f"\n✓ LSTM Model created")
        print(f"  Input shape: {input_shape}")
        print(f"  Output classes: {num_classes}")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model"""
        print("\n" + "="*70)
        print("COMPILING MODEL")
        print("="*70)
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print(f"\n✓ Model compiled")
        print(f"  Optimizer: Adam (lr={learning_rate})")
        print(f"  Loss: categorical_crossentropy")
        print(f"  Metrics: accuracy, precision, recall")
        
        # Print model summary
        print("\nModel Summary:")
        self.model.summary()
        
        # Count parameters
        trainable_params = self.model.count_params()
        print(f"\n✓ Trainable parameters: {trainable_params:,}")
    
    def setup_callbacks(self, model_name):
        """Setup training callbacks"""
        callback_list = [
            # Model checkpoint - save best model
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / f'{model_name}_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                str(self.log_dir / f'{model_name}_training.csv')
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=str(self.log_dir / model_name),
                histogram_freq=0
            )
        ]
        
        return callback_list
    
    def train(self, epochs=20, batch_size=64, learning_rate=0.001):
        """Train the model"""
        print("\n" + "="*70)
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print("="*70)
        
        # Load metadata
        self.load_metadata()
        
        # Prepare input shape
        n_features = self.metadata['n_features']
        n_classes = self.metadata['n_classes']
        
        if self.model_type == 'cnn':
            input_shape = (n_features, 1)
            self.model = self.create_cnn_model(input_shape, n_classes)
        elif self.model_type == 'lstm':
            timesteps = 11
            features_per_step = n_features // timesteps
            input_shape = (timesteps, features_per_step)
            self.model = self.create_lstm_model(input_shape, n_classes)
        
        # Compile model
        self.compile_model(learning_rate)
        
        # Setup callbacks
        model_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        callback_list = self.setup_callbacks(model_name)
        
        # Calculate steps
        train_size = self.metadata['train_size']
        val_size = self.metadata['val_size']
        
        steps_per_epoch = train_size // batch_size
        validation_steps = val_size // batch_size
        
        print(f"\n✓ Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        
        # Prepare data paths
        processed_dir = self.data_dir / 'processed'

        if self.use_subset:
            X_train_file = processed_dir / 'X_train_20pct.npy'
            y_train_file = processed_dir / 'y_train_20pct.npy'
            metadata_file = self.data_dir / 'metadata_20pct.pkl'
        else:
            X_train_file = processed_dir / 'X_train.npy'
            y_train_file = processed_dir / 'y_train.npy'
            metadata_file = self.data_dir / 'metadata.pkl'
        
        # Train model
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        self.history = self.model.fit(
            self.load_data_generator(
                X_train_file,  # ✅ Dùng biến này!
                y_train_file,  # ✅ Dùng biến này!
                batch_size=batch_size,
                shuffle=True
            ),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.load_data_generator(
                processed_dir / 'X_val.npy',
                processed_dir / 'y_val.npy',
                batch_size=batch_size,
                shuffle=False
            ),
            validation_steps=validation_steps,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE")
        print("="*70)
        
        # Save final model
        final_model_path = self.model_dir / f'{model_name}_final.keras'
        self.model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Save training history
        history_path = self.log_dir / f'{model_name}_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"✓ Training history saved to: {history_path}")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        print("\n" + "="*70)
        print("GENERATING TRAINING PLOTS")
        print("="*70)
        
        history = self.history.history
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_type.upper()} Training History', fontsize=16)
        
        # Plot 1: Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision
        axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Recall
        axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plot_dir / f'{self.model_type}_training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {plot_path}")
        plt.close()
    
    def evaluate(self, batch_size=64):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print(f"EVALUATING {self.model_type.upper()} MODEL")
        print("="*70)
        
        if self.model is None:
            print("No model available. Please train first.")
            return
        
        processed_dir = self.data_dir / 'processed'
        test_size = self.metadata['test_size']
        test_steps = test_size // batch_size
        
        # Evaluate
        print(f"\nEvaluating on test set ({test_size:,} samples)...")
        results = self.model.evaluate(
            self.load_data_generator(
                processed_dir / 'X_test.npy',
                processed_dir / 'y_test.npy',
                batch_size=batch_size,
                shuffle=False
            ),
            steps=test_steps,
            verbose=1
        )
        
        # Print results
        print("\n" + "="*70)
        print("TEST SET RESULTS")
        print("="*70)
        
        metrics = ['Loss', 'Accuracy', 'Precision', 'Recall']
        for metric, value in zip(metrics, results):
            print(f"  {metric}: {value:.4f}")
        
        # Calculate F1 Score
        precision = results[2]
        recall = results[3]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  F1 Score: {f1:.4f}")
        
        # Generate predictions for detailed analysis
        print("\n" + "="*70)
        print("GENERATING DETAILED PREDICTIONS")
        print("="*70)
        
        # Load test data
        X_test = np.load(processed_dir / 'X_test.npy')
        y_test = np.load(processed_dir / 'y_test.npy')
        
        # Reshape for model
        if self.model_type == 'cnn':
            X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
        elif self.model_type == 'lstm':
            timesteps = 11
            features_per_step = X_test.shape[1] // timesteps
            X_test_reshaped = X_test[:, :timesteps * features_per_step]
            X_test_reshaped = X_test_reshaped.reshape(-1, timesteps, features_per_step)
        
        # Predict in batches
        print(f"  Predicting {len(X_test):,} samples...")
        y_pred_proba = self.model.predict(X_test_reshaped, batch_size=batch_size, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_test, y_pred, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'{self.model_type.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = self.plot_dir / f'{self.model_type}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Save evaluation results
        eval_results = {
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_precision': float(results[2]),
            'test_recall': float(results[3]),
            'test_f1': float(f1),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        eval_path = self.log_dir / f'{self.model_type}_evaluation.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"✓ Evaluation results saved to: {eval_path}")
        
        # Clean up
        del X_test, X_test_reshaped, y_pred_proba, y_pred
        gc.collect()
        
        return eval_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN/LSTM for Network IDS')
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'lstm', 'both'],
                        help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='./split_data',
                        help='Directory with split data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing model')
    parser.add_argument('--use_subset', action='store_true',  # ✅ THÊM DÒNG NÀY
                        help='Use 20% subset for training')
    
    args = parser.parse_args()
    
    models_to_train = ['cnn', 'lstm'] if args.model == 'both' else [args.model]
    
    for model_type in models_to_train:
        print("\n" + "="*70)
        print(f"PROCESSING {model_type.upper()} MODEL")
        print("="*70)
        
        trainer = NetworkIDSTrainer(
            data_dir=args.data_dir,
            model_type=model_type,
            output_dir=args.output_dir,
            use_subset=args.use_subset
        )
        
        if not args.eval_only:
            # Train
            trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            # Plot training history
            trainer.plot_training_history()
        
        # Evaluate
        trainer.evaluate(batch_size=args.batch_size)
        
        # Clear memory
        del trainer
        gc.collect()
        tf.keras.backend.clear_session()
        
        print(f"\n✓ {model_type.upper()} model processing complete!")
    
    print("\n" + "="*70)
    print("✓ ALL MODELS PROCESSED SUCCESSFULLY!")
    print("="*70)


if __name__ == '__main__':
    main()