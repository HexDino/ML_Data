# 🚦 Traffic Sign Recognition - ResNet18 Custom

> **Branch:** `feature/resnet-18`  
> **Framework:** PyTorch  
> **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)

Dự án này xây dựng một hệ thống nhận diện biển báo giao thông (43 lớp) hiệu năng cao sử dụng kiến trúc **ResNet-18** tùy chỉnh. Dự án tập trung giải quyết các vấn đề thực tế như điều kiện ánh sáng kém (bằng CLAHE), mất cân bằng dữ liệu (bằng Focal Loss) và sai lệch mapping nhãn trong quá trình load dữ liệu.

---

## 💾 Download Model Checkpoint

Sử dụng checkpoint đã được Upload lên HuggingFace với name model là "khacdiep2208/my_custom_resnet18"


---

## 🌟 Key Features

* **ResNet-18 Backbone:** Sử dụng kiến trúc ResNet-18 gọn nhẹ, thay đổi lớp Fully Connected cuối cùng để phù hợp với 43 lớp biển báo.
* **Advanced Preprocessing (CLAHE):** Áp dụng *Contrast Limited Adaptive Histogram Equalization* trên không gian màu LAB. Kỹ thuật này giúp mô hình nhận diện tốt biển báo trong điều kiện thiếu sáng hoặc bị bóng đổ (vấn đề lớn nhất của GTSRB).
* **Focal Loss:** Thay thế CrossEntropyLoss truyền thống bằng Focal Loss để mô hình tập trung học các mẫu khó (hard mining) và giảm ảnh hưởng của mất cân bằng dữ liệu.
* **Robust Data Augmentation:** Sử dụng Random Rotation, Affine Translation và Color Jitter để tăng tính tổng quát hóa.
* **Production-Ready Evaluation:** Pipeline đánh giá tự động fix lỗi mapping index của `ImageFolder`, vẽ Confusion Matrix và tính toán Accuracy chính xác trên tập Test thực tế.

## 📂 Cấu trúc thư mục

```bash
Traffic-Sign-Recognition/
├── data/                       # Dữ liệu GTSRB
│   ├── Train/                  # Chứa 43 subfolders (0-42)
│   ├── Test/                   # Chứa ảnh Test (folder chứa ảnh .png)
│   ├── Train.csv               # Metadata & Ground Truth cho tập Train
│   └── Test.csv                # Metadata & Ground Truth cho tập Test
├── src/
│   ├── __init__.py
│   ├── resnet.py               # Định nghĩa kiến trúc Class ResNet18_Custom
│   ├── dataset.py              # Custom Dataset, DataLoader & Transforms (chứa logic CLAHE)
│   ├── train.py                # Script huấn luyện chính (Training Loop)
│   └── utils.py                # Các hàm phụ trợ (FocalLoss, save_checkpoint)
├── checkpoints/                # Nơi lưu model tốt nhất (.pth)
├── logs/                       # Lưu log training (csv) để vẽ biểu đồ loss
└── visualize.ipynb             # Notebook visualize data, infer, debug và đánh giá model
```

## ⚙️ Cài đặt & Yêu cầu
Đảm bảo bạn đã cài đặt Python và các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn sử dụng

### 1. Training

Chạy script ```train.py```. Script sẽ tự động chia tập Train/Val, áp dụng Augmentation, tính toán Loss và lưu model tốt nhất.

```bash
python train.py
```
* **Config mặc định:** Image Size 48x48, Batch Size 64, 30 Epochs, Learning Rate 1e-3.

* **Output:** Model sẽ được lưu tại `checkpoints/resnet18_custom_best.pth`.

### 2. Đánh giá & Visualize (Inference)
Sử dụng Jupyter Notebook `visualize.ipynb` để thực hiện các bước:

1. Load model và weights từ checkpoint.

2. Chạy dự đoán trên ảnh đơn lẻ.

3. Tính Accuracy trên toàn bộ 12,630 ảnh Test.

4. Vẽ Confusion Matrix để phân tích các lỗi sai cụ thể.

## 🧠 Chi tiết kỹ thuật quan trọng

### 1. Pipeline Tiền xử lý (Preprocessing)
Sự thành công của model phụ thuộc lớn vào sự đồng nhất giữa lúc Train và Test:

* **Resize:** Cố định kích thước 48x48 pixels.

* **CLAHE:** Tăng cường độ tương phản cục bộ.

* **Normalization:** Sử dụng Mean/Std chuẩn của ImageNet: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

### 2. Xử lý lỗi Class Mapping (Kinh nghiệm thực tế)

Khi sử dụng `torchvision.datasets.ImageFolder`, thứ tự class được đánh số dựa trên tên thư mục (Alphabetical sort).

* **Vấn đề:** Folder tên "10" sẽ đứng trước "2" (theo thứ tự từ điển). Do đó, Model học rằng biển báo lớp 10 có index là 2.

* **Giải pháp:** Trong pipeline đánh giá (`visualize.ipynb`), hệ thống tự động tạo một từ điển `idx_to_class` để map ngược lại kết quả dự đoán về đúng ID thực tế trong file `Test.csv`, đảm bảo Accuracy chính xác > 95%.

## 📊 Kết quả (Performance)
* **Model Architecture:** ResNet-18 Custom.

* **Best Validation Accuracy:** ~98-99% (trong quá trình train).

* **Test Accuracy (Real-world):** >95% (Sau khi áp dụng CLAHE và fix mapping).

