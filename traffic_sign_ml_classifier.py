"""
Traffic Sign Classification using Traditional Machine Learning
==============================================================
GTSRB Dataset - German Traffic Sign Recognition Benchmark
Models: SVM, KNN, Random Forest with HOG Feature Extraction

Author: ML Project
University Project: Comparing ML vs Deep Learning for Traffic Sign Recognition
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Image processing
import cv2
from skimage.feature import hog
from skimage import exposure

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration class for the pipeline."""
    
    # Dataset paths
    BASE_PATH = Path("archive")
    TRAIN_CSV = BASE_PATH / "Train.csv"
    TEST_CSV = BASE_PATH / "Test.csv"
    META_CSV = BASE_PATH / "Meta.csv"
    
    # Image preprocessing
    IMG_SIZE = (32, 32)  # Resize images to 32x32
    
    # HOG parameters
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Number of classes
    NUM_CLASSES = 43
    
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
                print(f"  Warning: Class folder {class_id} not found")
                continue
            
            class_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            
            for img_path in class_images:
                try:
                    img = self.load_image(img_path)
                    images.append(img)
                    labels.append(class_id)
                except Exception as e:
                    pass  # Skip problematic images
            
            if class_id % 10 == 0:
                print(f"  Progress: Class {class_id}/42 loaded...")
        
        print(f"  Loaded {len(images)} images successfully!")
        return np.array(images), np.array(labels)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class FeatureExtractor:
    """Handles HOG feature extraction from images."""
    
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
    
    def extract_features_batch(self, images: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Extract HOG features from a batch of images."""
        if verbose:
            print(f"\nExtracting HOG features from {len(images)} images...")
        
        features = []
        for idx, img in enumerate(images):
            if verbose and idx % 5000 == 0 and idx > 0:
                print(f"  Progress: {idx}/{len(images)} features extracted...")
            
            feat = self.extract_hog_features(img)
            features.append(feat)
        
        features = np.array(features)
        if verbose:
            print(f"  Feature extraction complete!")
            print(f"  Feature vector shape: {features.shape}")
        
        return features
    
    def visualize_hog(self, image: np.ndarray, title: str = "HOG Visualization"):
        """Visualize HOG features for a single image."""
        features, hog_image = hog(
            image,
            orientations=self.config.HOG_ORIENTATIONS,
            pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
            cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True
        )
        
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(hog_image_rescaled, cmap='gray')
        axes[1].set_title('HOG Features')
        axes[1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('hog_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("HOG visualization saved to 'hog_visualization.png'")


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
    
    def get_models(self) -> Dict[str, Any]:
        """Define the models to be trained."""
        models = {
            'SVM': SVC(
                kernel='rbf',
                random_state=self.config.RANDOM_STATE,
                probability=True
            ),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for tuning."""
        param_grids = {
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        }
        return param_grids
    
    def train_model(
        self, 
        name: str, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        tune_hyperparameters: bool = False,
        param_grid: Dict = None
    ) -> Tuple[Any, float]:
        """Train a single model and return it with training time."""
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        if tune_hyperparameters and param_grid:
            print(f"  Performing GridSearchCV for hyperparameter tuning...")
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, train_time
        else:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            return model, train_time
    
    def evaluate_model(
        self, 
        name: str, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate a trained model and return metrics."""
        print(f"\nEvaluating {name}...")
        
        # Inference time measurement
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
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
            'inference_time': inference_time,
            'per_sample_time_ms': per_sample_time,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        return results
    
    def train_and_evaluate_all(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Dict]:
        """Train and evaluate all models."""
        # Scale features
        print("\nScaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = self.get_models()
        param_grids = self.get_param_grids() if tune_hyperparameters else {}
        
        all_results = {}
        
        for name, model in models.items():
            # Train
            trained_model, train_time = self.train_model(
                name, model, X_train_scaled, y_train,
                tune_hyperparameters=tune_hyperparameters,
                param_grid=param_grids.get(name)
            )
            
            # Evaluate
            results = self.evaluate_model(name, trained_model, X_test_scaled, y_test)
            results['train_time'] = train_time
            results['model'] = trained_model
            
            all_results[name] = results
            self.models[name] = trained_model
            
            # Print results
            print(f"\n  {name} Results:")
            print(f"  {'─'*40}")
            print(f"  Accuracy:      {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"  Precision:     {results['precision']:.4f}")
            print(f"  Recall:        {results['recall']:.4f}")
            print(f"  F1-Score:      {results['f1_score']:.4f}")
            print(f"  Train Time:    {train_time:.2f} seconds")
            print(f"  Inference Time: {results['inference_time']:.4f} seconds (total)")
            print(f"  Per-sample:    {results['per_sample_time_ms']:.4f} ms")
        
        self.results = all_results
        return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def plot_sample_images(
        self, 
        images: np.ndarray, 
        labels: np.ndarray, 
        num_samples: int = 16
    ):
        """Plot sample images from the dataset."""
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        # Select random samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for idx, ax in enumerate(axes.flat):
            img_idx = indices[idx]
            ax.imshow(images[img_idx], cmap='gray')
            class_name = self.config.CLASS_NAMES.get(labels[img_idx], f"Class {labels[img_idx]}")
            ax.set_title(f"Class {labels[img_idx]}\n{class_name[:20]}...", fontsize=8)
            ax.axis('off')
        
        plt.suptitle('Sample Images from Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Sample images saved to 'sample_images.png'")
    
    def plot_class_distribution(self, labels: np.ndarray, title: str = "Class Distribution"):
        """Plot the distribution of classes in the dataset."""
        plt.figure(figsize=(14, 6))
        
        unique, counts = np.unique(labels, return_counts=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
        bars = plt.bar(unique, counts, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Class ID', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(unique, fontsize=8)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(count), ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Class distribution saved to 'class_distribution.png'")
    
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
            annot=False,  # Too many classes for annotations
            fmt='.2f',
            cmap='Blues',
            xticklabels=range(self.config.NUM_CLASSES),
            yticklabels=range(self.config.NUM_CLASSES)
        )
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}\n(Normalized)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to 'confusion_matrix_{model_name.lower().replace(' ', '_')}.png'")
    
    def plot_model_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of all models."""
        model_names = list(results.keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Colors for each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        accuracies = [results[m]['accuracy'] * 100 for m in model_names]
        bars1 = ax1.bar(model_names, accuracies, color=colors, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim([0, 100])
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        f1_scores = [results[m]['f1_score'] * 100 for m in model_names]
        bars2 = ax2.bar(model_names, f1_scores, color=colors, edgecolor='black')
        ax2.set_ylabel('F1-Score (%)')
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_ylim([0, 100])
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{f1:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Training time comparison
        ax3 = axes[1, 0]
        train_times = [results[m]['train_time'] for m in model_names]
        bars3 = ax3.bar(model_names, train_times, color=colors, edgecolor='black')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time Comparison')
        for bar, t in zip(bars3, train_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. Inference time comparison
        ax4 = axes[1, 1]
        inference_times = [results[m]['per_sample_time_ms'] for m in model_names]
        bars4 = ax4.bar(model_names, inference_times, color=colors, edgecolor='black')
        ax4.set_ylabel('Time (ms per sample)')
        ax4.set_title('Inference Time Comparison')
        for bar, t in zip(bars4, inference_times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{t:.4f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Machine Learning Models Comparison - GTSRB Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Model comparison saved to 'model_comparison.png'")
    
    def plot_all_metrics(self, results: Dict[str, Dict]):
        """Plot all metrics in a comprehensive view."""
        model_names = list(results.keys())
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (metric, key) in enumerate(zip(metrics, metric_keys)):
            values = [results[m][key] * 100 for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor='black')
        
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('All Metrics Comparison by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('all_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("All metrics comparison saved to 'all_metrics_comparison.png'")


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

def print_results_summary(results: Dict[str, Dict], config: Config):
    """Print a comprehensive summary of all results."""
    print("\n")
    print("=" * 80)
    print("                    RESULTS SUMMARY - GTSRB Classification")
    print("=" * 80)
    
    # Create results DataFrame
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy (%)': f"{res['accuracy']*100:.2f}",
            'Precision (%)': f"{res['precision']*100:.2f}",
            'Recall (%)': f"{res['recall']*100:.2f}",
            'F1-Score (%)': f"{res['f1_score']*100:.2f}",
            'Train Time (s)': f"{res['train_time']:.2f}",
            'Inference (ms/sample)': f"{res['per_sample_time_ms']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy'] * 100
    
    print("\n" + "-" * 80)
    print(f"🏆 BEST PERFORMING MODEL: {best_model}")
    print(f"   Accuracy: {best_accuracy:.2f}%")
    print("-" * 80)
    
    # Save results to CSV
    df_summary.to_csv('results_summary.csv', index=False)
    print("\n📊 Results saved to 'results_summary.csv'")
    
    return best_model


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(use_csv: bool = True, tune_hyperparameters: bool = False, use_test_set: bool = True):
    """
    Main pipeline for traffic sign classification.
    
    Args:
        use_csv: If True, load data from CSV files. If False, load from folder structure.
        tune_hyperparameters: If True, perform GridSearchCV for hyperparameter tuning.
        use_test_set: If True, use the separate test set. If False, split train data.
    """
    print("=" * 80)
    print("   GTSRB Traffic Sign Classification - Traditional ML Pipeline")
    print("   Models: SVM (RBF), KNN, Random Forest")
    print("   Features: HOG (Histogram of Oriented Gradients)")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    
    # Initialize components
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    model_trainer = ModelTrainer(config)
    visualizer = Visualizer(config)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    if use_csv:
        # Load from CSV files
        X_train_images, y_train = data_loader.load_dataset_from_csv(config.TRAIN_CSV)
        
        if use_test_set:
            X_test_images, y_test = data_loader.load_dataset_from_csv(config.TEST_CSV, is_test=True)
        else:
            # Split training data
            X_train_images, X_test_images, y_train, y_test = train_test_split(
                X_train_images, y_train, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE,
                stratify=y_train
            )
    else:
        # Load from folder structure
        train_dir = config.BASE_PATH / "Train"
        X_images, y_labels = data_loader.load_dataset_from_folders(train_dir)
        
        # Split into train/test
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
    
    # Visualize sample images and class distribution
    print("\n📈 Generating visualizations...")
    visualizer.plot_sample_images(X_train_images, y_train)
    visualizer.plot_class_distribution(y_train, "Training Set Class Distribution")
    
    # =========================================================================
    # STEP 2: Feature Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Feature Extraction (HOG)")
    print("=" * 60)
    
    # Visualize HOG for a sample image
    sample_idx = np.random.randint(0, len(X_train_images))
    sample_class = y_train[sample_idx]
    feature_extractor.visualize_hog(
        X_train_images[sample_idx], 
        f"HOG Features - Class {sample_class}: {config.CLASS_NAMES.get(sample_class, 'Unknown')[:30]}"
    )
    
    # Extract features
    X_train_features = feature_extractor.extract_features_batch(X_train_images)
    X_test_features = feature_extractor.extract_features_batch(X_test_images)
    
    print(f"\n📊 Feature Statistics:")
    print(f"   Training features shape: {X_train_features.shape}")
    print(f"   Testing features shape:  {X_test_features.shape}")
    print(f"   Features per image:      {X_train_features.shape[1]}")
    
    # =========================================================================
    # STEP 3: Model Training and Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Model Training and Evaluation")
    print("=" * 60)
    
    results = model_trainer.train_and_evaluate_all(
        X_train_features, y_train,
        X_test_features, y_test,
        tune_hyperparameters=tune_hyperparameters
    )
    
    # =========================================================================
    # STEP 4: Results Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Results Visualization")
    print("=" * 60)
    
    # Print summary and get best model
    best_model_name = print_results_summary(results, config)
    
    # Plot comparisons
    visualizer.plot_model_comparison(results)
    visualizer.plot_all_metrics(results)
    
    # Plot confusion matrix for best model
    best_results = results[best_model_name]
    visualizer.plot_confusion_matrix(
        best_results['y_test'], 
        best_results['y_pred'], 
        best_model_name
    )
    
    # Print detailed classification report for best model
    print(f"\n📋 Detailed Classification Report - {best_model_name}:")
    print("-" * 60)
    print(classification_report(
        best_results['y_test'], 
        best_results['y_pred'],
        target_names=[f"Class {i}" for i in range(config.NUM_CLASSES)],
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("                    PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   - sample_images.png")
    print("   - class_distribution.png")
    print("   - hog_visualization.png")
    print("   - model_comparison.png")
    print("   - all_metrics_comparison.png")
    print(f"   - confusion_matrix_{best_model_name.lower().replace(' ', '_')}.png")
    print("   - results_summary.csv")
    
    return results, model_trainer.models


# =============================================================================
# QUICK EVALUATION MODE (Using subset for faster testing)
# =============================================================================

def quick_evaluation(sample_size: int = 5000):
    """
    Quick evaluation mode using a subset of data for faster testing.
    Useful for debugging and initial testing.
    """
    print("=" * 80)
    print("   QUICK EVALUATION MODE (Subset of Data)")
    print("=" * 80)
    
    config = Config()
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    model_trainer = ModelTrainer(config)
    
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
    
    # Train and evaluate
    results = model_trainer.train_and_evaluate_all(
        X_train_feat, y_train, X_test_feat, y_test, 
        tune_hyperparameters=False
    )
    
    print_results_summary(results, config)
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GTSRB Traffic Sign Classification using Traditional ML'
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Run quick evaluation with subset of data'
    )
    parser.add_argument(
        '--tune', 
        action='store_true', 
        help='Enable hyperparameter tuning with GridSearchCV'
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
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_evaluation(sample_size=args.sample_size)
    else:
        results, models = main(
            use_csv=not args.use_folders,
            tune_hyperparameters=args.tune,
            use_test_set=not args.split_train
        )

