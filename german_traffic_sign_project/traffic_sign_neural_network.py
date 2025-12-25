"""
Traffic Sign Classification using Neural Network (MLP)
=======================================================
GTSRB Dataset - German Traffic Sign Recognition Benchmark
Model: Multi-Layer Perceptron (Neural Network)

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
from typing import Tuple, Dict, Any, List

# Image processing
from skimage.feature import hog
from skimage import exposure

# Machine Learning - Neural Network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for the Neural Network pipeline."""
    
    # Dataset paths
    BASE_PATH = Path("archive")
    TRAIN_CSV = BASE_PATH / "Train.csv"
    TEST_CSV = BASE_PATH / "Test.csv"
    META_CSV = BASE_PATH / "Meta.csv"
    
    # Image preprocessing
    IMG_SIZE = (32, 32)  # Resize images to 32x32
    
    # HOG parameters (for feature extraction)
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Number of classes
    NUM_CLASSES = 43
    
    # Neural Network configurations to compare
    # Each tuple represents (hidden_layer_sizes, name, description)
    NN_CONFIGS = [
        ((128,), "NN_1Layer_128", "1 Hidden Layer (128 neurons)"),
        ((256, 128), "NN_2Layer_256_128", "2 Hidden Layers (256, 128)"),
        ((512, 256, 128), "NN_3Layer_512_256_128", "3 Hidden Layers (512, 256, 128)"),
        ((256, 128, 64), "NN_3Layer_256_128_64", "3 Hidden Layers (256, 128, 64)"),
    ]
    
    # Class names for GTSRB (German Traffic Signs)
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
# DATA LOADING
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of the GTSRB dataset."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fixed size
        img = cv2.resize(img, self.config.IMG_SIZE)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize pixel values to [0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def load_dataset_from_csv(self, csv_path: Path, is_test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from CSV file."""
        print(f"\nLoading dataset from: {csv_path}")
        
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        
        total = len(df)
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"  Progress: {idx}/{total} images loaded...")
            
            img_path = self.config.BASE_PATH / row['Path']
            try:
                img = self.load_image(img_path)
                images.append(img)
                labels.append(row['ClassId'])
            except Exception as e:
                print(f"  Warning: Could not load {img_path}: {e}")
        
        print(f"  Loaded {len(images)} images successfully!")
        return np.array(images), np.array(labels)
    
    def load_dataset_from_folders(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from folder structure (root/class_id/images)."""
        print(f"\nLoading dataset from folders: {data_dir}")
        
        images = []
        labels = []
        
        for class_id in range(self.config.NUM_CLASSES):
            class_dir = data_dir / str(class_id)
            if not class_dir.exists():
                print(f"  Warning: Class directory {class_dir} does not exist")
                continue
            
            class_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            
            for img_path in class_images:
                try:
                    img = self.load_image(img_path)
                    images.append(img)
                    labels.append(class_id)
                except Exception as e:
                    pass
            
            if class_id % 10 == 0:
                print(f"  Loaded class {class_id}...")
        
        print(f"  Loaded {len(images)} images successfully!")
        return np.array(images), np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Handles feature extraction from images."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from a single grayscale image."""
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
    
    def extract_pixel_features(self, image: np.ndarray) -> np.ndarray:
        """Flatten image pixels as features (alternative to HOG)."""
        return image.flatten()
    
    def extract_features_batch(
        self, 
        images: np.ndarray, 
        method: str = 'hog',
        verbose: bool = True
    ) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            images: Array of images
            method: 'hog' for HOG features, 'pixel' for raw pixels
            verbose: Whether to print progress
        """
        if verbose:
            print(f"\nExtracting {method.upper()} features from {len(images)} images...")
        
        features = []
        for idx, img in enumerate(images):
            if verbose and idx % 5000 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(images)} features extracted...")
            
            if method == 'hog':
                feat = self.extract_hog_features(img)
            else:
                feat = self.extract_pixel_features(img)
            features.append(feat)
        
        features = np.array(features)
        if verbose:
            print(f"  Feature extraction complete!")
            print(f"  Feature vector shape: {features.shape}")
        
        return features


# =============================================================================
# NEURAL NETWORK IMPLEMENTATION
# =============================================================================

class NeuralNetworkTrainer:
    """
    Handles Neural Network training and evaluation.
    
    Mạng nơ-ron nhân tạo (Artificial Neural Network - ANN):
    --------------------------------------------------------
    
    1. CẤU TRÚC MẠNG:
       - Input Layer: Nhận dữ liệu đầu vào (features)
       - Hidden Layers: Các lớp ẩn xử lý thông tin
       - Output Layer: Đưa ra kết quả dự đoán
    
    2. CÔNG THỨC TÍNH TOÁN:
       
       Đối với mỗi neuron:
       z = Σ(wi * xi) + b
       a = activation(z)
       
       Trong đó:
       - xi: input từ neuron trước
       - wi: trọng số (weight)
       - b: bias
       - z: tổng có trọng số
       - a: giá trị kích hoạt (activation)
    
    3. HÀM KÍCH HOẠT (Activation Functions):
       - ReLU: f(x) = max(0, x)
       - Sigmoid: f(x) = 1 / (1 + e^(-x))
       - Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
       - Softmax: f(xi) = e^xi / Σ(e^xj) (cho output layer)
    
    4. THUẬT TOÁN HỌC (Backpropagation):
       - Forward pass: Tính output
       - Backward pass: Tính gradient của loss
       - Update weights: w = w - learning_rate * gradient
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.training_history = {}
    
    def create_neural_network(
        self,
        hidden_layer_sizes: Tuple[int, ...],
        activation: str = 'relu',
        solver: str = 'adam',
        max_iter: int = 300,
        learning_rate_init: float = 0.001,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> MLPClassifier:
        """
        Create a Multi-Layer Perceptron (Neural Network) classifier.
        
        Args:
            hidden_layer_sizes: Tuple defining number of neurons in each hidden layer
                                e.g., (256, 128) = 2 hidden layers with 256 and 128 neurons
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Optimization algorithm ('adam', 'sgd', 'lbfgs')
            max_iter: Maximum number of iterations
            learning_rate_init: Initial learning rate
            early_stopping: Whether to use early stopping
            verbose: Whether to print training progress
        
        Returns:
            MLPClassifier: Configured neural network
        """
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            learning_rate='adaptive',
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.config.RANDOM_STATE,
            verbose=verbose
        )
        return model
    
    def train_neural_network(
        self,
        name: str,
        model: MLPClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[MLPClassifier, float, Dict]:
        """
        Train a neural network model.
        
        Args:
            name: Model name for identification
            model: The MLPClassifier to train
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Tuple of (trained model, training time, training info)
        """
        print(f"\n{'='*60}")
        print(f"Training Neural Network: {name}")
        print(f"{'='*60}")
        print(f"  Architecture: Input({X_train.shape[1]}) -> {model.hidden_layer_sizes} -> Output({self.config.NUM_CLASSES})")
        print(f"  Activation: {model.activation}")
        print(f"  Solver: {model.solver}")
        print(f"  Learning Rate: {model.learning_rate_init}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Training information
        training_info = {
            'n_iter': model.n_iter_,
            'loss': model.loss_,
            'best_loss': model.best_loss_ if hasattr(model, 'best_loss_') else model.loss_,
            'n_layers': model.n_layers_,
            'n_outputs': model.n_outputs_,
            'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else []
        }
        
        print(f"\n  ✓ Training completed!")
        print(f"  Iterations: {training_info['n_iter']}")
        print(f"  Final Loss: {training_info['loss']:.6f}")
        print(f"  Training Time: {train_time:.2f} seconds")
        
        return model, train_time, training_info
    
    def evaluate_model(
        self,
        name: str,
        model: MLPClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate a trained neural network and return metrics."""
        print(f"\nEvaluating {name}...")
        
        # Inference time measurement
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Probability predictions (for confidence analysis)
        y_proba = model.predict_proba(X_test)
        confidence = np.max(y_proba, axis=1).mean()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-sample inference time
        per_sample_time = (inference_time / len(X_test)) * 1000  # in milliseconds
        
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
        configs: List[Tuple] = None
    ) -> Dict[str, Dict]:
        """Train and evaluate multiple neural network configurations."""
        
        # Scale features
        print("\nScaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if configs is None:
            configs = self.config.NN_CONFIGS
        
        all_results = {}
        
        for hidden_layers, name, description in configs:
            print(f"\n{'#'*60}")
            print(f"# Configuration: {description}")
            print(f"{'#'*60}")
            
            # Create model
            model = self.create_neural_network(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=300,
                learning_rate_init=0.001,
                early_stopping=True,
                verbose=False
            )
            
            # Train
            trained_model, train_time, training_info = self.train_neural_network(
                name, model, X_train_scaled, y_train
            )
            
            # Evaluate
            results = self.evaluate_model(name, trained_model, X_test_scaled, y_test)
            results['train_time'] = train_time
            results['training_info'] = training_info
            results['model'] = trained_model
            results['description'] = description
            results['hidden_layers'] = hidden_layers
            
            all_results[name] = results
            self.models[name] = trained_model
            self.training_history[name] = training_info.get('loss_curve', [])
            
            # Print results
            print(f"\n  📊 {name} Results:")
            print(f"  {'─'*40}")
            print(f"  Accuracy:       {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"  Precision:      {results['precision']:.4f}")
            print(f"  Recall:         {results['recall']:.4f}")
            print(f"  F1-Score:       {results['f1_score']:.4f}")
            print(f"  Confidence:     {results['confidence']:.4f}")
            print(f"  Train Time:     {train_time:.2f} seconds")
            print(f"  Inference Time: {results['inference_time']:.4f} seconds (total)")
            print(f"  Per-sample:     {results['per_sample_time_ms']:.4f} ms")
        
        self.results = all_results
        return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

class NeuralNetworkVisualizer:
    """Handles visualization for Neural Network analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_network_architecture(
        self,
        input_size: int,
        hidden_layers: Tuple[int, ...],
        output_size: int,
        name: str = "Neural Network"
    ):
        """Visualize neural network architecture."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Layer sizes
        layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        n_layers = len(layer_sizes)
        
        # Positions
        layer_positions = np.linspace(0, 1, n_layers)
        max_neurons = max(min(size, 10) for size in layer_sizes)  # Cap at 10 for display
        
        # Colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # Draw neurons and connections
        for layer_idx, (layer_size, layer_x) in enumerate(zip(layer_sizes, layer_positions)):
            # Limit display neurons
            display_size = min(layer_size, 10)
            neuron_positions = np.linspace(0.1, 0.9, display_size)
            
            color = colors[layer_idx % len(colors)]
            
            for neuron_idx, neuron_y in enumerate(neuron_positions):
                # Draw neuron
                circle = plt.Circle((layer_x, neuron_y), 0.02, 
                                   color=color, ec='black', lw=2, zorder=2)
                ax.add_patch(circle)
                
                # Draw connections to next layer
                if layer_idx < n_layers - 1:
                    next_display_size = min(layer_sizes[layer_idx + 1], 10)
                    next_positions = np.linspace(0.1, 0.9, next_display_size)
                    for next_y in next_positions:
                        ax.plot([layer_x, layer_positions[layer_idx + 1]], 
                               [neuron_y, next_y], 
                               'gray', alpha=0.2, lw=0.5, zorder=1)
            
            # Layer label
            layer_name = ['Input', 'Output'][0] if layer_idx == 0 else \
                        ['Input', 'Output'][1] if layer_idx == n_layers - 1 else \
                        f'Hidden {layer_idx}'
            
            ax.text(layer_x, -0.05, f'{layer_name}\n({layer_size})', 
                   ha='center', va='top', fontsize=10, fontweight='bold')
            
            # Show "..." if truncated
            if layer_size > 10:
                ax.text(layer_x, 0.5, '...', ha='center', va='center', 
                       fontsize=14, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Neural Network Architecture: {name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'nn_architecture_{name.lower().replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Architecture visualization saved!")
    
    def plot_training_curves(self, training_history: Dict[str, List[float]]):
        """Plot training loss curves for all models."""
        plt.figure(figsize=(12, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (name, loss_curve) in enumerate(training_history.items()):
            if len(loss_curve) > 0:
                plt.plot(loss_curve, label=name, color=colors[idx % len(colors)], lw=2)
        
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curves - Neural Networks', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nn_training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Training curves saved to 'nn_training_curves.png'")
    
    def plot_model_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of all neural network configurations."""
        model_names = [results[m]['description'] for m in results.keys()]
        model_keys = list(results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Colors for each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. Accuracy comparison
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
        
        # 2. F1-Score comparison
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
        
        # 3. Training time comparison
        ax3 = axes[1, 0]
        train_times = [results[m]['train_time'] for m in model_keys]
        bars3 = ax3.bar(range(len(model_names)), train_times, color=colors, edgecolor='black')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time Comparison')
        for bar, t in zip(bars3, train_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 4. Number of iterations
        ax4 = axes[1, 1]
        iterations = [results[m]['training_info']['n_iter'] for m in model_keys]
        bars4 = ax4.bar(range(len(model_names)), iterations, color=colors, edgecolor='black')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax4.set_ylabel('Iterations')
        ax4.set_title('Training Iterations')
        for bar, it in zip(bars4, iterations):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{it}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Neural Network Configurations Comparison - GTSRB Dataset', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('nn_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Model comparison saved to 'nn_model_comparison.png'")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ):
        """Plot confusion matrix for a model."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(16, 14))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
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
        plt.savefig(f'nn_confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved!")
    
    def plot_confidence_distribution(self, results: Dict[str, Dict]):
        """Plot confidence distribution for predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (name, res) in enumerate(results.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            y_proba = res['y_proba']
            max_probs = np.max(y_proba, axis=1)
            
            # Correct and incorrect predictions
            correct_mask = res['y_pred'] == res['y_test']
            
            ax.hist(max_probs[correct_mask], bins=50, alpha=0.7, 
                   label='Correct', color='green')
            ax.hist(max_probs[~correct_mask], bins=50, alpha=0.7, 
                   label='Incorrect', color='red')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title(f'{res["description"]}')
            ax.legend()
        
        plt.suptitle('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('nn_confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Confidence distribution saved to 'nn_confidence_distribution.png'")


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(results: Dict[str, Dict], config: Config):
    """Print a comprehensive summary of all Neural Network results."""
    print("\n")
    print("=" * 80)
    print("          NEURAL NETWORK RESULTS SUMMARY - GTSRB Classification")
    print("=" * 80)
    
    # Create results DataFrame
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            'Model': res['description'],
            'Architecture': str(res['hidden_layers']),
            'Accuracy (%)': f"{res['accuracy']*100:.2f}",
            'Precision (%)': f"{res['precision']*100:.2f}",
            'Recall (%)': f"{res['recall']*100:.2f}",
            'F1-Score (%)': f"{res['f1_score']*100:.2f}",
            'Confidence': f"{res['confidence']:.4f}",
            'Train Time (s)': f"{res['train_time']:.2f}",
            'Iterations': res['training_info']['n_iter']
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy'] * 100
    best_desc = results[best_model]['description']
    
    print("\n" + "-" * 80)
    print(f"🏆 BEST PERFORMING NEURAL NETWORK: {best_desc}")
    print(f"   Architecture: {results[best_model]['hidden_layers']}")
    print(f"   Accuracy: {best_accuracy:.2f}%")
    print("-" * 80)
    
    # Save results to CSV
    df_summary.to_csv('nn_results_summary.csv', index=False)
    print("\n📊 Results saved to 'nn_results_summary.csv'")
    
    return best_model


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(
    use_csv: bool = True,
    use_test_set: bool = True,
    feature_method: str = 'hog'
):
    """
    Main pipeline for Neural Network traffic sign classification.
    
    Args:
        use_csv: If True, load data from CSV files.
        use_test_set: If True, use separate test set.
        feature_method: 'hog' for HOG features, 'pixel' for raw pixels
    """
    print("=" * 80)
    print("   GTSRB Traffic Sign Classification - Neural Network Pipeline")
    print("   Model: Multi-Layer Perceptron (MLP)")
    print("   Features: HOG (Histogram of Oriented Gradients)")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    nn_trainer = NeuralNetworkTrainer(config)
    visualizer = NeuralNetworkVisualizer(config)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset")
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
    else:
        train_dir = config.BASE_PATH / "Train"
        X_images, y_labels = data_loader.load_dataset_from_folders(train_dir)
        
        X_train_images, X_test_images, y_train, y_test = train_test_split(
            X_images, y_labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_labels
        )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(X_train_images)}")
    print(f"   Testing samples:  {len(X_test_images)}")
    print(f"   Image shape:      {X_train_images[0].shape}")
    print(f"   Number of classes: {len(np.unique(y_train))}")
    
    # =========================================================================
    # STEP 2: Feature Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Feature Extraction")
    print("=" * 60)
    
    X_train_features = feature_extractor.extract_features_batch(
        X_train_images, method=feature_method
    )
    X_test_features = feature_extractor.extract_features_batch(
        X_test_images, method=feature_method
    )
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Training features shape: {X_train_features.shape}")
    print(f"   Testing features shape:  {X_test_features.shape}")
    print(f"   Features per image:      {X_train_features.shape[1]}")
    
    # Visualize architecture for the first config
    first_config = config.NN_CONFIGS[0]
    visualizer.plot_network_architecture(
        input_size=X_train_features.shape[1],
        hidden_layers=first_config[0],
        output_size=config.NUM_CLASSES,
        name=first_config[1]
    )
    
    # =========================================================================
    # STEP 3: Neural Network Training and Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Neural Network Training and Evaluation")
    print("=" * 60)
    
    results = nn_trainer.train_and_evaluate_all(
        X_train_features, y_train,
        X_test_features, y_test
    )
    
    # =========================================================================
    # STEP 4: Results Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Results Visualization")
    print("=" * 60)
    
    # Print summary and get best model
    best_model_name = print_results_summary(results, config)
    
    # Plot visualizations
    visualizer.plot_training_curves(nn_trainer.training_history)
    visualizer.plot_model_comparison(results)
    visualizer.plot_confidence_distribution(results)
    
    # Confusion matrix for best model
    best_results = results[best_model_name]
    visualizer.plot_confusion_matrix(
        best_results['y_test'],
        best_results['y_pred'],
        best_results['description']
    )
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report - {best_results['description']}:")
    print("-" * 60)
    print(classification_report(
        best_results['y_test'],
        best_results['y_pred'],
        target_names=[f"Class {i}" for i in range(config.NUM_CLASSES)],
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("                    NEURAL NETWORK PIPELINE COMPLETED!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   - nn_architecture_*.png")
    print("   - nn_training_curves.png")
    print("   - nn_model_comparison.png")
    print("   - nn_confidence_distribution.png")
    print("   - nn_confusion_matrix_*.png")
    print("   - nn_results_summary.csv")
    
    return results, nn_trainer.models


# =============================================================================
# QUICK EVALUATION MODE
# =============================================================================

def quick_evaluation(sample_size: int = 5000):
    """Quick evaluation mode using a subset of data."""
    print("=" * 80)
    print("   NEURAL NETWORK - QUICK EVALUATION MODE")
    print("=" * 80)
    
    config = Config()
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    nn_trainer = NeuralNetworkTrainer(config)
    
    # Load data
    X_images, y_labels = data_loader.load_dataset_from_csv(config.TRAIN_CSV)
    
    # Take subset
    if len(X_images) > sample_size:
        indices = np.random.choice(len(X_images), sample_size, replace=False)
        X_images = X_images[indices]
        y_labels = y_labels[indices]
    
    # Split data
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_images, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"\n📊 Quick Mode - Using {len(X_train_img)} training, {len(X_test_img)} testing samples")
    
    # Extract features
    X_train_feat = feature_extractor.extract_features_batch(X_train_img)
    X_test_feat = feature_extractor.extract_features_batch(X_test_img)
    
    # Use only 2 configs for quick mode
    quick_configs = [
        ((128,), "NN_1Layer_128", "1 Hidden Layer (128)"),
        ((256, 128), "NN_2Layer_256_128", "2 Hidden Layers (256, 128)"),
    ]
    
    # Train and evaluate
    results = nn_trainer.train_and_evaluate_all(
        X_train_feat, y_train, X_test_feat, y_test,
        configs=quick_configs
    )
    
    print_results_summary(results, config)
    
    return results


# =============================================================================
# SINGLE MODEL TRAINING (Customizable)
# =============================================================================

def train_custom_neural_network(
    hidden_layers: Tuple[int, ...] = (256, 128),
    activation: str = 'relu',
    learning_rate: float = 0.001,
    max_iterations: int = 300
):
    """
    Train a single custom neural network configuration.
    
    Args:
        hidden_layers: Tuple of neurons per hidden layer
        activation: Activation function ('relu', 'tanh', 'logistic')
        learning_rate: Initial learning rate
        max_iterations: Maximum training iterations
    """
    print("=" * 80)
    print("   CUSTOM NEURAL NETWORK TRAINING")
    print("=" * 80)
    print(f"\n  Configuration:")
    print(f"    Hidden Layers: {hidden_layers}")
    print(f"    Activation: {activation}")
    print(f"    Learning Rate: {learning_rate}")
    print(f"    Max Iterations: {max_iterations}")
    
    config = Config()
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    
    # Load and prepare data
    X_train_images, y_train = data_loader.load_dataset_from_csv(config.TRAIN_CSV)
    X_test_images, y_test = data_loader.load_dataset_from_csv(config.TEST_CSV, is_test=True)
    
    X_train_feat = feature_extractor.extract_features_batch(X_train_images)
    X_test_feat = feature_extractor.extract_features_batch(X_test_images)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    
    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver='adam',
        max_iter=max_iterations,
        learning_rate_init=learning_rate,
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=config.RANDOM_STATE,
        verbose=True
    )
    
    print("\nTraining...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"  Training completed in {train_time:.2f} seconds")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Iterations: {model.n_iter_}")
    print(f"{'='*60}")
    
    return model, accuracy


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='GTSRB Traffic Sign Classification using Neural Networks'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation with subset of data'
    )
    parser.add_argument(
        '--use-folders',
        action='store_true',
        help='Load data from folder structure instead of CSV'
    )
    parser.add_argument(
        '--split-train',
        action='store_true',
        help='Split training data instead of using separate test set'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Sample size for quick evaluation mode (default: 5000)'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='hog',
        choices=['hog', 'pixel'],
        help='Feature extraction method (default: hog)'
    )
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Train a single custom neural network'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='256,128',
        help='Hidden layer sizes for custom training (comma-separated, e.g., "256,128,64")'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_evaluation(sample_size=args.sample_size)
    elif args.custom:
        layers = tuple(int(x) for x in args.layers.split(','))
        model, accuracy = train_custom_neural_network(hidden_layers=layers)
    else:
        results, models = main(
            use_csv=not args.use_folders,
            use_test_set=not args.split_train,
            feature_method=args.features
        )
