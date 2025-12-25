"""
Traffic Sign Classification using PyTorch Neural Network (GPU Accelerated)
===========================================================================
GTSRB Dataset - German Traffic Sign Recognition Benchmark
Model: Multi-Layer Perceptron (MLP) with PyTorch
GPU Support: NVIDIA CUDA (RTX 3070 and similar)

Author: ML Project
University Project: Comparing ML vs Deep Learning for Traffic Sign Recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import warnings
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Scikit-learn for metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Image processing
from skimage.feature import hog

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# GPU/DEVICE CONFIGURATION
# =============================================================================

def get_device():
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU Detected: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("⚠️ No GPU detected, using CPU")
    return device


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for the PyTorch Neural Network pipeline."""
    
    # Dataset paths
    BASE_PATH = Path("archive")
    TRAIN_CSV = BASE_PATH / "Train.csv"
    TEST_CSV = BASE_PATH / "Test.csv"
    META_CSV = BASE_PATH / "Meta.csv"
    
    # Image preprocessing
    IMG_SIZE = (32, 32)
    
    # HOG parameters
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # PyTorch training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    
    # Number of classes
    NUM_CLASSES = 43
    
    # Neural Network configurations to compare
    NN_CONFIGS = [
        ((128,), "PyTorch_1Layer_128", "1 Hidden Layer (128)"),
        ((256, 128), "PyTorch_2Layer_256_128", "2 Hidden Layers (256, 128)"),
        ((512, 256, 128), "PyTorch_3Layer_512_256_128", "3 Hidden Layers (512, 256, 128)"),
        ((256, 128, 64), "PyTorch_3Layer_256_128_64", "3 Hidden Layers (256, 128, 64)"),
    ]
    
    # Class names for GTSRB
    CLASS_NAMES = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }


# =============================================================================
# DATASET CLASS
# =============================================================================

class TrafficSignDataset(Dataset):
    """PyTorch Dataset for Traffic Sign data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Numpy array of features (N, D)
            labels: Numpy array of labels (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =============================================================================
# DATA LOADING
# =============================================================================

class ImageDataLoader:
    """Handles loading and preprocessing of the GTSRB dataset."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.config.IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def load_dataset_from_csv(self, csv_path: Path, is_test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV file."""
        print(f"\n📂 Loading dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        
        total = len(df)
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"   Progress: {idx}/{total} images loaded...")
            
            img_path = self.config.BASE_PATH / row['Path']
            try:
                img = self.load_image(img_path)
                images.append(img)
                labels.append(row['ClassId'])
            except Exception as e:
                pass
        
        print(f"   ✅ Loaded {len(images)} images successfully!")
        return np.array(images), np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Handles HOG feature extraction."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from a single image."""
        features = hog(
            image,
            orientations=self.config.HOG_ORIENTATIONS,
            pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
            cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return features
    
    def extract_features_batch(self, images: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Extract HOG features from a batch of images."""
        if verbose:
            print(f"\n🔍 Extracting HOG features from {len(images)} images...")
        
        features = []
        for idx, img in enumerate(images):
            if verbose and idx % 5000 == 0 and idx > 0:
                print(f"   Progress: {idx}/{len(images)} features extracted...")
            features.append(self.extract_hog_features(img))
        
        features = np.array(features)
        if verbose:
            print(f"   ✅ Feature extraction complete! Shape: {features.shape}")
        
        return features


# =============================================================================
# NEURAL NETWORK MODELS (PyTorch)
# =============================================================================

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for classification.
    
    Kiến trúc:
    - Input layer: Nhận features
    - Hidden layers: ReLU activation + Dropout
    - Output layer: Softmax (implicit in CrossEntropyLoss)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: Tuple[int, ...], 
        num_classes: int,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: Tuple of hidden layer sizes, e.g., (256, 128)
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(MLPClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_architecture_string(self) -> str:
        """Get human-readable architecture string."""
        arch = f"Input({self.input_size})"
        for h in self.hidden_sizes:
            arch += f" → Dense({h}) → BN → ReLU → Dropout"
        arch += f" → Output({self.num_classes})"
        return arch


# =============================================================================
# TRAINING ENGINE
# =============================================================================

class TrainingEngine:
    """Handles model training and evaluation with GPU support."""
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.models = {}
        self.results = {}
        self.training_history = {}
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 10,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict]:
        """
        Train a PyTorch model with early stopping.
        
        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            verbose: Print training progress
        
        Returns:
            Tuple of (trained model, training history)
        """
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item() * batch_features.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"   Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"   ⚡ Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        train_time = time.time() - start_time
        history['train_time'] = train_time
        history['final_epoch'] = epoch + 1
        
        if verbose:
            print(f"   ✅ Training completed in {train_time:.2f} seconds")
            print(f"   Best Validation Loss: {best_val_loss:.4f}")
        
        return model, history
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        name: str
    ) -> Dict[str, Any]:
        """Evaluate a trained model on test data."""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                
                outputs = model(batch_features)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        inference_time = time.time() - start_time
        
        y_pred = np.array(all_predictions)
        y_test = np.array(all_labels)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        confidence = np.max(y_proba, axis=1).mean()
        
        per_sample_time = (inference_time / len(y_test)) * 1000  # ms
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confidence': confidence,
            'inference_time': inference_time,
            'per_sample_time_ms': per_sample_time,
            'y_pred': y_pred,
            'y_test': y_test,
            'y_proba': y_proba
        }
        
        return results
    
    def train_and_evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        configs: List[Tuple] = None,
        batch_size: int = 128
    ) -> Dict[str, Dict]:
        """Train and evaluate multiple neural network configurations."""
        
        # Scale features
        print("\n📊 Scaling features with StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Split training into train/val
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train,
            test_size=0.1,
            random_state=self.config.RANDOM_STATE,
            stratify=y_train
        )
        
        # Create datasets and dataloaders
        train_dataset = TrafficSignDataset(X_train_split, y_train_split)
        val_dataset = TrafficSignDataset(X_val_split, y_val_split)
        test_dataset = TrafficSignDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        
        if configs is None:
            configs = self.config.NN_CONFIGS
        
        input_size = X_train_scaled.shape[1]
        all_results = {}
        
        for hidden_layers, name, description in configs:
            print(f"\n{'='*60}")
            print(f"🧠 Training: {description}")
            print(f"{'='*60}")
            
            # Create model
            model = MLPClassifier(
                input_size=input_size,
                hidden_sizes=hidden_layers,
                num_classes=self.config.NUM_CLASSES,
                dropout_rate=0.3
            )
            
            print(f"   Architecture: {model.get_architecture_string()}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Device: {self.device}")
            
            # Train
            trained_model, history = self.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.config.NUM_EPOCHS,
                learning_rate=self.config.LEARNING_RATE,
                patience=self.config.EARLY_STOPPING_PATIENCE,
                verbose=True
            )
            
            # Evaluate
            results = self.evaluate_model(trained_model, test_loader, name)
            results['train_time'] = history['train_time']
            results['training_history'] = history
            results['model'] = trained_model
            results['description'] = description
            results['hidden_layers'] = hidden_layers
            
            all_results[name] = results
            self.models[name] = trained_model
            self.training_history[name] = history
            
            # Print results
            print(f"\n   📊 {name} Results:")
            print(f"   {'─'*40}")
            print(f"   Accuracy:       {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"   Precision:      {results['precision']:.4f}")
            print(f"   Recall:         {results['recall']:.4f}")
            print(f"   F1-Score:       {results['f1_score']:.4f}")
            print(f"   Confidence:     {results['confidence']:.4f}")
            print(f"   Train Time:     {results['train_time']:.2f}s")
            print(f"   Inference:      {results['per_sample_time_ms']:.4f} ms/sample")
        
        self.results = all_results
        return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

class PyTorchVisualizer:
    """Visualization for PyTorch Neural Network results."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_training_curves(self, training_histories: Dict[str, Dict]):
        """Plot training and validation loss/accuracy curves."""
        n_models = len(training_histories)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, 4*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (name, history) in enumerate(training_histories.items()):
            # Loss plot
            axes[idx, 0].plot(history['train_loss'], label='Train Loss', color=colors[0], lw=2)
            axes[idx, 0].plot(history['val_loss'], label='Val Loss', color=colors[1], lw=2)
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Loss')
            axes[idx, 0].set_title(f'{name} - Loss')
            axes[idx, 0].legend()
            axes[idx, 0].grid(alpha=0.3)
            
            # Accuracy plot
            train_acc = [a * 100 for a in history['train_acc']]
            val_acc = [a * 100 for a in history['val_acc']]
            axes[idx, 1].plot(train_acc, label='Train Acc', color=colors[0], lw=2)
            axes[idx, 1].plot(val_acc, label='Val Acc', color=colors[1], lw=2)
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('Accuracy (%)')
            axes[idx, 1].set_title(f'{name} - Accuracy')
            axes[idx, 1].legend()
            axes[idx, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pytorch_nn_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📈 Training curves saved to 'pytorch_nn_training_curves.png'")
    
    def plot_model_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of all models."""
        model_names = [results[m]['description'] for m in results.keys()]
        model_keys = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. Accuracy
        ax1 = axes[0, 0]
        accuracies = [results[m]['accuracy'] * 100 for m in model_keys]
        bars1 = ax1.bar(range(len(model_names)), accuracies, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim([0, 100])
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 2. F1-Score
        ax2 = axes[0, 1]
        f1_scores = [results[m]['f1_score'] * 100 for m in model_keys]
        bars2 = ax2.bar(range(len(model_names)), f1_scores, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax2.set_ylabel('F1-Score (%)')
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_ylim([0, 100])
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{f1:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Training Time
        ax3 = axes[1, 0]
        train_times = [results[m]['train_time'] for m in model_keys]
        bars3 = ax3.bar(range(len(model_names)), train_times, color=colors, edgecolor='black')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time Comparison (GPU)')
        for bar, t in zip(bars3, train_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 4. Inference Time
        ax4 = axes[1, 1]
        inference_times = [results[m]['per_sample_time_ms'] for m in model_keys]
        bars4 = ax4.bar(range(len(model_names)), inference_times, color=colors, edgecolor='black')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax4.set_ylabel('Time (ms per sample)')
        ax4.set_title('Inference Time Comparison')
        for bar, t in zip(bars4, inference_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{t:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('PyTorch Neural Network Comparison - GTSRB Dataset (GPU)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('pytorch_nn_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Model comparison saved to 'pytorch_nn_model_comparison.png'")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt='.2f',
            cmap='Blues',
            xticklabels=range(self.config.NUM_CLASSES),
            yticklabels=range(self.config.NUM_CLASSES)
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}\n(Normalized)', fontsize=14)
        plt.tight_layout()
        
        filename = f'pytorch_nn_confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Confusion matrix saved!")


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(results: Dict[str, Dict], device: torch.device):
    """Print comprehensive results summary."""
    print("\n")
    print("=" * 80)
    print("       PyTorch NEURAL NETWORK RESULTS - GTSRB Classification")
    print(f"                    Device: {device}")
    print("=" * 80)
    
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            'Model': res['description'],
            'Architecture': str(res['hidden_layers']),
            'Accuracy (%)': f"{res['accuracy']*100:.2f}",
            'Precision (%)': f"{res['precision']*100:.2f}",
            'Recall (%)': f"{res['recall']*100:.2f}",
            'F1-Score (%)': f"{res['f1_score']*100:.2f}",
            'Train Time (s)': f"{res['train_time']:.2f}",
            'Inference (ms)': f"{res['per_sample_time_ms']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy'] * 100
    best_desc = results[best_model]['description']
    
    print("\n" + "-" * 80)
    print(f"🏆 BEST PERFORMING MODEL: {best_desc}")
    print(f"   Architecture: {results[best_model]['hidden_layers']}")
    print(f"   Accuracy: {best_accuracy:.2f}%")
    print("-" * 80)
    
    df_summary.to_csv('pytorch_nn_results_summary.csv', index=False)
    print("\n📊 Results saved to 'pytorch_nn_results_summary.csv'")
    
    return best_model


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(use_csv: bool = True, use_test_set: bool = True):
    """Main pipeline for PyTorch Neural Network training."""
    
    print("=" * 80)
    print("   🚀 GTSRB Traffic Sign Classification - PyTorch Neural Network")
    print("   GPU Accelerated Training with CUDA")
    print("=" * 80)
    
    # Get device
    device = get_device()
    
    # Initialize
    config = Config()
    data_loader = ImageDataLoader(config)
    feature_extractor = FeatureExtractor(config)
    trainer = TrainingEngine(config, device)
    visualizer = PyTorchVisualizer(config)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("📂 STEP 1: Loading Dataset")
    print("=" * 60)
    
    if use_csv:
        X_train_images, y_train = data_loader.load_dataset_from_csv(config.TRAIN_CSV)
        
        if use_test_set:
            X_test_images, y_test = data_loader.load_dataset_from_csv(config.TEST_CSV, is_test=True)
        else:
            X_train_images, X_test_images, y_train, y_test = train_test_split(
                X_train_images, y_train,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=y_train
            )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(X_train_images)}")
    print(f"   Testing samples:  {len(X_test_images)}")
    print(f"   Number of classes: {len(np.unique(y_train))}")
    
    # =========================================================================
    # STEP 2: Feature Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔍 STEP 2: Feature Extraction (HOG)")
    print("=" * 60)
    
    X_train_features = feature_extractor.extract_features_batch(X_train_images)
    X_test_features = feature_extractor.extract_features_batch(X_test_images)
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Training features: {X_train_features.shape}")
    print(f"   Testing features:  {X_test_features.shape}")
    
    # =========================================================================
    # STEP 3: Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("🧠 STEP 3: Neural Network Training (PyTorch + GPU)")
    print("=" * 60)
    
    results = trainer.train_and_evaluate_all(
        X_train_features, y_train,
        X_test_features, y_test,
        batch_size=config.BATCH_SIZE
    )
    
    # =========================================================================
    # STEP 4: Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("📈 STEP 4: Results Visualization")
    print("=" * 60)
    
    best_model_name = print_results_summary(results, device)
    
    visualizer.plot_training_curves(trainer.training_history)
    visualizer.plot_model_comparison(results)
    
    best_results = results[best_model_name]
    visualizer.plot_confusion_matrix(
        best_results['y_test'],
        best_results['y_pred'],
        best_results['description']
    )
    
    # Classification report
    print(f"\n📋 Classification Report - {best_results['description']}:")
    print("-" * 60)
    print(classification_report(
        best_results['y_test'],
        best_results['y_pred'],
        target_names=[f"Class {i}" for i in range(config.NUM_CLASSES)],
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("              ✅ PyTorch PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   - pytorch_nn_training_curves.png")
    print("   - pytorch_nn_model_comparison.png")
    print("   - pytorch_nn_confusion_matrix_*.png")
    print("   - pytorch_nn_results_summary.csv")
    
    return results, trainer.models


# =============================================================================
# QUICK EVALUATION
# =============================================================================

def quick_evaluation(sample_size: int = 5000):
    """Quick evaluation with subset of data."""
    
    print("=" * 80)
    print("   ⚡ PyTorch Neural Network - QUICK EVALUATION MODE")
    print("=" * 80)
    
    device = get_device()
    config = Config()
    data_loader = ImageDataLoader(config)
    feature_extractor = FeatureExtractor(config)
    trainer = TrainingEngine(config, device)
    
    # Load data
    X_images, y_labels = data_loader.load_dataset_from_csv(config.TRAIN_CSV)
    
    # Take subset
    if len(X_images) > sample_size:
        indices = np.random.choice(len(X_images), sample_size, replace=False)
        X_images = X_images[indices]
        y_labels = y_labels[indices]
    
    # Split
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_images, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"\n📊 Quick Mode - {len(X_train_img)} training, {len(X_test_img)} testing samples")
    
    # Extract features
    X_train_feat = feature_extractor.extract_features_batch(X_train_img)
    X_test_feat = feature_extractor.extract_features_batch(X_test_img)
    
    # Quick configs
    quick_configs = [
        ((128,), "PyTorch_1Layer_128", "1 Hidden Layer (128)"),
        ((256, 128), "PyTorch_2Layer_256_128", "2 Hidden Layers (256, 128)"),
    ]
    
    # Train
    results = trainer.train_and_evaluate_all(
        X_train_feat, y_train, X_test_feat, y_test,
        configs=quick_configs,
        batch_size=64
    )
    
    print_results_summary(results, device)
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='GTSRB Traffic Sign Classification - PyTorch Neural Network (GPU)'
    )
    parser.add_argument('--quick', action='store_true', help='Quick evaluation mode')
    parser.add_argument('--sample-size', type=int, default=5000, help='Sample size for quick mode')
    parser.add_argument('--split-train', action='store_true', help='Split train data instead of using test set')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config if needed
    if args.epochs != 50:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size != 128:
        Config.BATCH_SIZE = args.batch_size
    if args.lr != 0.001:
        Config.LEARNING_RATE = args.lr
    
    if args.quick:
        results = quick_evaluation(sample_size=args.sample_size)
    else:
        results, models = main(
            use_csv=True,
            use_test_set=not args.split_train
        )
