"""
Traffic Sign Prediction - Dự đoán biển báo giao thông từ ảnh bất kỳ
===================================================================
Sử dụng model đã được train để dự đoán biển báo giao thông

Usage:
    python predict_traffic_sign.py --image <đường_dẫn_ảnh>
    python predict_traffic_sign.py --image path/to/image.jpg
    python predict_traffic_sign.py --train  # Train và lưu model mới
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
from skimage.feature import hog
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CẤU HÌNH
# =============================================================================

class Config:
    IMG_SIZE = (32, 32)
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    NUM_CLASSES = 43
    
    # Đường dẫn lưu model
    PROJECT_DIR = Path(__file__).parent
    MODEL_PATH = PROJECT_DIR / "models" / "traffic_sign_model.pth"
    SCALER_PATH = PROJECT_DIR / "models" / "traffic_sign_scaler.npy"
    
    # Tên các lớp biển báo
    CLASS_NAMES = {
        0: 'Giới hạn tốc độ (20km/h)',
        1: 'Giới hạn tốc độ (30km/h)',
        2: 'Giới hạn tốc độ (50km/h)',
        3: 'Giới hạn tốc độ (60km/h)',
        4: 'Giới hạn tốc độ (70km/h)',
        5: 'Giới hạn tốc độ (80km/h)',
        6: 'Hết giới hạn tốc độ (80km/h)',
        7: 'Giới hạn tốc độ (100km/h)',
        8: 'Giới hạn tốc độ (120km/h)',
        9: 'Cấm vượt',
        10: 'Cấm vượt xe trên 3.5 tấn',
        11: 'Đường ưu tiên tại ngã tư tiếp theo',
        12: 'Đường ưu tiên',
        13: 'Nhường đường',
        14: 'Dừng lại',
        15: 'Cấm xe',
        16: 'Cấm xe trên 3.5 tấn',
        17: 'Cấm đi ngược chiều',
        18: 'Chú ý nguy hiểm',
        19: 'Đường cong nguy hiểm bên trái',
        20: 'Đường cong nguy hiểm bên phải',
        21: 'Đường cong kép',
        22: 'Đường gồ ghề',
        23: 'Đường trơn',
        24: 'Đường hẹp bên phải',
        25: 'Công trường',
        26: 'Đèn giao thông',
        27: 'Người đi bộ',
        28: 'Trẻ em qua đường',
        29: 'Xe đạp qua đường',
        30: 'Cẩn thận băng/tuyết',
        31: 'Động vật hoang dã',
        32: 'Hết tất cả giới hạn',
        33: 'Rẽ phải phía trước',
        34: 'Rẽ trái phía trước',
        35: 'Chỉ đi thẳng',
        36: 'Đi thẳng hoặc rẽ phải',
        37: 'Đi thẳng hoặc rẽ trái',
        38: 'Đi bên phải',
        39: 'Đi bên trái',
        40: 'Bắt buộc đi vòng xuyến',
        41: 'Hết cấm vượt',
        42: 'Hết cấm vượt xe trên 3.5 tấn'
    }


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron cho phân loại biển báo."""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def preprocess_image(image_path, config):
    """Tiền xử lý ảnh đầu vào."""
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Chuyển sang RGB và resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE)
    
    # Chuyển sang grayscale và normalize
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    
    return gray, img


def extract_hog_features(image, config):
    """Trích xuất đặc trưng HOG từ ảnh."""
    features = hog(
        image,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features


# =============================================================================
# PREDICTION
# =============================================================================

def load_model(config, device):
    """Load model đã được train."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model tại: {config.MODEL_PATH}\n"
            "Hãy chạy 'python predict_traffic_sign.py --train' trước để train model."
        )
    
    # Load checkpoint
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    
    # Tạo model với cùng cấu hình
    model = MLPClassifier(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes'],
        num_classes=config.NUM_CLASSES
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load scaler parameters
    scaler_mean = checkpoint['scaler_mean']
    scaler_scale = checkpoint['scaler_scale']
    
    print(f"✅ Đã load model từ: {config.MODEL_PATH}")
    print(f"   Accuracy khi train: {checkpoint['accuracy']*100:.2f}%")
    
    return model, scaler_mean, scaler_scale


def predict_single_image(image_path, model, scaler_mean, scaler_scale, config, device):
    """Dự đoán một ảnh duy nhất."""
    # Tiền xử lý ảnh
    gray_img, color_img = preprocess_image(image_path, config)
    
    # Trích xuất features
    features = extract_hog_features(gray_img, config)
    
    # Chuẩn hóa features
    features = (features - scaler_mean) / scaler_scale
    
    # Chuyển sang tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Lấy top 3 dự đoán
    top3_probs, top3_classes = torch.topk(probabilities, 3)
    top3_results = []
    for prob, cls in zip(top3_probs[0], top3_classes[0]):
        top3_results.append({
            'class_id': cls.item(),
            'class_name': config.CLASS_NAMES[cls.item()],
            'confidence': prob.item()
        })
    
    return {
        'predicted_class': predicted_class,
        'class_name': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'top3': top3_results,
        'color_image': color_img
    }


def display_prediction(result, image_path):
    """Hiển thị kết quả dự đoán."""
    print("\n" + "=" * 60)
    print("🚦 KẾT QUẢ DỰ ĐOÁN BIỂN BÁO GIAO THÔNG")
    print("=" * 60)
    print(f"\n📷 Ảnh: {image_path}")
    print(f"\n🎯 Kết quả: {result['class_name']}")
    print(f"   Class ID: {result['predicted_class']}")
    print(f"   Độ tin cậy: {result['confidence']*100:.2f}%")
    
    print(f"\n📊 Top 3 dự đoán:")
    for i, pred in enumerate(result['top3'], 1):
        print(f"   {i}. {pred['class_name']} ({pred['confidence']*100:.2f}%)")
    
    print("=" * 60)
    
    # Lưu ảnh kết quả
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(result['color_image'])
        ax.set_title(f"Dự đoán: {result['class_name']}\n"
                     f"Độ tin cậy: {result['confidence']*100:.1f}%", 
                     fontsize=14)
        ax.axis('off')
        
        output_path = "prediction_result.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"\n📁 Đã lưu kết quả vào: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Không thể lưu ảnh kết quả: {e}")


# =============================================================================
# TRAINING & SAVE MODEL
# =============================================================================

def train_and_save_model():
    """Train model và lưu lại để sử dụng sau."""
    print("=" * 60)
    print("🔧 TRAINING MODEL VÀ LƯU LẠI")
    print("=" * 60)
    
    # Import các module cần thiết
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")
    
    # Load data
    BASE_PATH = Path("archive")
    TRAIN_CSV = BASE_PATH / "Train.csv"
    
    print(f"\n📂 Loading dữ liệu từ: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"   Progress: {idx}/{len(df)} images...")
        
        img_path = BASE_PATH / row['Path']
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, config.IMG_SIZE)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                gray = gray.astype(np.float32) / 255.0
                images.append(gray)
                labels.append(row['ClassId'])
        except:
            pass
    
    print(f"   ✅ Loaded {len(images)} images!")
    
    # Extract HOG features
    print(f"\n🔍 Extracting HOG features...")
    features = []
    for idx, img in enumerate(images):
        if idx % 5000 == 0 and idx > 0:
            print(f"   Progress: {idx}/{len(images)}")
        features.append(extract_hog_features(img, config))
    
    X = np.array(features)
    y = np.array(labels)
    print(f"   ✅ Features shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    hidden_sizes = (512, 256, 128)  # Best architecture
    
    model = MLPClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=config.NUM_CLASSES
    ).to(device)
    
    print(f"\n🧠 Training Neural Network...")
    print(f"   Architecture: {hidden_sizes}")
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    num_epochs = 50
    best_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}] - Accuracy: {acc*100:.2f}%")
    
    print(f"\n✅ Training hoàn tất! Best Accuracy: {best_acc*100:.2f}%")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'accuracy': best_acc,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, config.MODEL_PATH)
    
    print(f"💾 Đã lưu model vào: {config.MODEL_PATH}")
    print(f"\n🎉 Giờ bạn có thể chạy:")
    print(f"   python predict_traffic_sign.py --image <đường_dẫn_ảnh>")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dự đoán biển báo giao thông từ ảnh'
    )
    parser.add_argument('--image', '-i', type=str, help='Đường dẫn đến ảnh cần dự đoán')
    parser.add_argument('--train', action='store_true', help='Train và lưu model mới')
    
    args = parser.parse_args()
    
    if args.train:
        train_and_save_model()
        return
    
    if not args.image:
        parser.print_help()
        print("\n" + "=" * 60)
        print("VÍ DỤ SỬ DỤNG:")
        print("=" * 60)
        print("\n1. Train model (chạy 1 lần đầu):")
        print("   python predict_traffic_sign.py --train")
        print("\n2. Dự đoán ảnh:")
        print("   python predict_traffic_sign.py --image path/to/image.jpg")
        print("   python predict_traffic_sign.py -i archive/Test/00000.png")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy ảnh: {args.image}")
        return
    
    # Load model and predict
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    try:
        model, scaler_mean, scaler_scale = load_model(config, device)
        result = predict_single_image(
            args.image, model, scaler_mean, scaler_scale, config, device
        )
        display_prediction(result, args.image)
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
