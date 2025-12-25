# Traffic Sign Classification - Traditional Machine Learning

## GTSRB Dataset - German Traffic Sign Recognition Benchmark

A comprehensive implementation of traditional machine learning models (SVM, KNN, Random Forest) for traffic sign classification using HOG (Histogram of Oriented Gradients) features.

## Project Overview

This project is part of a university comparison study between **Traditional Machine Learning** and **Deep Learning** approaches for traffic sign recognition.

### Features
- **Data Preprocessing**: Image resizing, grayscale conversion, normalization
- **Feature Extraction**: HOG (Histogram of Oriented Gradients) using scikit-image
- **Models Implemented**:
  - Support Vector Machine (SVM) with RBF kernel
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier
- **Hyperparameter Tuning**: Optional GridSearchCV support
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Training/Inference time
- **Visualizations**: Confusion matrices, model comparisons, class distributions

## Dataset

The **GTSRB (German Traffic Sign Recognition Benchmark)** dataset contains:
- **43 classes** of traffic signs
- **~39,000+ training images**
- **~12,600+ test images**
- Variable image sizes (resized to 32x32 for this project)

### Dataset Structure
```
archive/
├── Train/
│   ├── 0/          # Class 0 images
│   ├── 1/          # Class 1 images
│   └── ...         # Classes 0-42
├── Test/
│   └── *.png       # Test images
├── Meta/
│   └── *.png       # Class representative images
├── Train.csv       # Training metadata
├── Test.csv        # Test metadata
└── Meta.csv        # Class metadata
```

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Basic Run (Full Pipeline)
```bash
python traffic_sign_ml_classifier.py
```

### Quick Evaluation (Faster, uses subset)
```bash
python traffic_sign_ml_classifier.py --quick
```

### With Hyperparameter Tuning
```bash
python traffic_sign_ml_classifier.py --tune
```

### All Command Line Options
```bash
python traffic_sign_ml_classifier.py --help
```

| Option | Description |
|--------|-------------|
| `--quick` | Run quick evaluation with subset of data (5000 samples by default) |
| `--tune` | Enable hyperparameter tuning with GridSearchCV |
| `--use-folders` | Load data from folder structure instead of CSV |
| `--split-train` | Split training data instead of using separate test set |
| `--sample-size N` | Sample size for quick evaluation mode (default: 5000) |

### Examples
```bash
# Quick test with 3000 samples
python traffic_sign_ml_classifier.py --quick --sample-size 3000

# Full pipeline with hyperparameter tuning
python traffic_sign_ml_classifier.py --tune

# Use folder structure and split training data
python traffic_sign_ml_classifier.py --use-folders --split-train
```

## Pipeline Steps

### 1. Data Loading
- Reads images from CSV file paths or folder structure
- Resizes images to 32x32 pixels
- Converts to grayscale
- Normalizes pixel values to [0, 1]

### 2. Feature Extraction (HOG)
- **Orientations**: 9 bins
- **Pixels per Cell**: 8x8
- **Cells per Block**: 2x2
- **Block Normalization**: L2-Hys

HOG features capture edge orientations and gradients, making them robust for traffic sign recognition.

### 3. Model Training
- **SVM (RBF)**: Support Vector Machine with Radial Basis Function kernel
- **KNN**: K-Nearest Neighbors (default k=5)
- **Random Forest**: Ensemble of decision trees

### 4. Evaluation
- Train/Test split (80/20) or use separate test set
- Metrics: Accuracy, Precision, Recall, F1-Score
- Training and inference time measurement

### 5. Visualization
- Sample images display
- Class distribution plots
- Model comparison charts
- Confusion matrices

## Output Files

After running the pipeline, the following files are generated:

| File | Description |
|------|-------------|
| `sample_images.png` | Sample images from the dataset |
| `class_distribution.png` | Bar chart of class frequencies |
| `hog_visualization.png` | HOG feature visualization |
| `model_comparison.png` | Bar charts comparing models |
| `all_metrics_comparison.png` | All metrics side-by-side |
| `confusion_matrix_*.png` | Confusion matrix for best model |
| `results_summary.csv` | Tabular results summary |

## Expected Results

Typical accuracy ranges for this dataset with HOG features:
- **SVM (RBF)**: 85-92%
- **KNN**: 80-88%
- **Random Forest**: 82-90%

*Note: Results may vary based on random state and hyperparameters.*

## Code Structure

```
traffic_sign_ml_classifier.py
├── Config                 # Configuration class
├── DataLoader             # Data loading and preprocessing
├── FeatureExtractor       # HOG feature extraction
├── ModelTrainer           # Model training and evaluation
├── Visualizer             # Visualization utilities
├── print_results_summary  # Results summary function
├── main                   # Main pipeline
├── quick_evaluation       # Quick evaluation mode
└── __main__               # CLI entry point
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **opencv-python**: Image loading and preprocessing
- **scikit-image**: HOG feature extraction
- **scikit-learn**: ML models and evaluation
- **matplotlib**: Plotting
- **seaborn**: Enhanced visualizations

## Tips for Better Results

1. **Increase image size** (e.g., 64x64) for more HOG features
2. **Enable hyperparameter tuning** (`--tune`) for optimal parameters
3. **Use data augmentation** if extending the project
4. **Try different HOG parameters** in the Config class
5. **Experiment with feature scaling** methods

## License

This project is for educational purposes as part of a university machine learning course.

## Author

ML Project - University Course on Machine Learning
