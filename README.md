# 🚦 Traffic Sign Classification Project

Machine Learning project để nhận dạng biển báo giao thông sử dụng Neural Network.

## 📁 Cấu trúc thư mục

```
ML_Data/
│
├── 📂 docs/                              # Tài liệu
│   ├── README.md                         # Hướng dẫn cũ
│   └── NEURAL_NETWORK_GUIDE.md           # Hướng dẫn Neural Network
│
├── 📂 setup/                             # Cài đặt môi trường
│   ├── environment.yml                   # Conda environment
│   ├── requirements.txt                  # Python packages
│   └── Anaconda3-*.sh                    # Anaconda installer
│
├── 📂 german_dataset/                    # Dataset biển báo Đức (GTSRB)
│   ├── Train/                            # Ảnh training (43 classes)
│   ├── Test/                             # Ảnh testing
│   ├── Meta/                             # Ảnh metadata
│   └── *.csv                             # Labels
│
├── 📂 german_traffic_sign_project/       # 🇩🇪 Project biển báo Đức
│   ├── traffic_sign_neural_network_pytorch.py  # Training với PyTorch (GPU)
│   ├── traffic_sign_neural_network.py    # Training không GPU
│   ├── traffic_sign_ml_classifier.py     # ML truyền thống (SVM, RF, etc.)
│   ├── predict_traffic_sign.py           # Dự đoán ảnh
│   ├── models/                           # Model đã train
│   │   └── traffic_sign_model.pth
│   ├── results/                          # Kết quả & biểu đồ
│   │   ├── pytorch_nn_training_curves.png
│   │   ├── confusion_matrix_svm.png
│   │   └── ...
│   └── test_thuc_te/                     # Ảnh test thực tế
│
├── 📂 vietnam_dataset/                   # Dataset biển báo Việt Nam
│   └── archive/
│       ├── images/                       # 3216 ảnh JPG
│       ├── labels/                       # Labels YOLO format
│       ├── classes_vie.txt               # Tên biển báo tiếng Việt
│       └── ...
│
└── 📂 vietnam_traffic_sign_project/      # 🇻🇳 Project biển báo Việt Nam
    ├── train_vietnam_traffic_sign.py     # Training với HOG + Color
    ├── predict_vietnam_traffic_sign.py   # Dự đoán ảnh
    ├── models/                           # Model đã train
    │   └── vietnam_traffic_sign_model.pth
    └── results/                          # Kết quả & biểu đồ
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── sample_predictions.png
        └── ...
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Tạo conda environment
cd setup
conda env create -f environment.yml
conda activate traffic_sign_nn
```

### 2. 🇩🇪 Project biển báo Đức (GTSRB)

```bash
cd german_traffic_sign_project

# Training
python traffic_sign_neural_network_pytorch.py

# Dự đoán
python predict_traffic_sign.py --image path/to/image.jpg
```

### 3. 🇻🇳 Project biển báo Việt Nam

```bash
cd vietnam_traffic_sign_project

# Training
python train_vietnam_traffic_sign.py

# Dự đoán
python predict_vietnam_traffic_sign.py --image path/to/image.jpg
```

## 📊 So sánh hai project

| Tính năng | German (GTSRB) | Vietnam |
|-----------|----------------|---------|
| **Số class** | 43 | 50 |
| **Số ảnh** | ~50,000 | ~8,000 |
| **Features** | HOG only | HOG + Color (RGB/HSV) |
| **Accuracy** | ~96% | ~96.5% |
| **GPU** | ✅ RTX 3070 | ✅ RTX 3070 |

## 🛠️ Công nghệ sử dụng

- **Framework**: PyTorch (GPU accelerated)
- **Features**: HOG (Histogram of Oriented Gradients) + Color Histogram
- **Model**: Multi-Layer Perceptron (MLP)
- **GPU**: NVIDIA RTX 3070 Laptop GPU

## 👨‍💻 Author

ML Project - HUST (Hanoi University of Science and Technology)
