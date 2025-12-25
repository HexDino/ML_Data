# 📚 GIẢI THÍCH ĐẦY ĐỦ CODE NHẬN DIỆN BIỂN BÁO GIAO THÔNG
## Tài liệu Chi tiết cho Người Thuyết trình

---

## 📋 MỤC LỤC

1. [Tổng quan Dự án](#1-tổng-quan-dự-án)
2. [Pipeline Xử lý Dữ liệu](#2-pipeline-xử-lý-dữ-liệu)
3. [Trích xuất Đặc trưng HOG](#3-trích-xuất-đặc-trưng-hog)
4. [Kiến trúc Neural Network](#4-kiến-trúc-neural-network)
5. [Các Layer Chi tiết](#5-các-layer-chi-tiết)
6. [Quá trình Training](#6-quá-trình-training)
7. [Các Kỹ thuật Tối ưu](#7-các-kỹ-thuật-tối-ưu)
8. [Công thức Toán học](#8-công-thức-toán-học)
9. [Giải thích từng Phần Code](#9-giải-thích-từng-phần-code)
10. [Câu hỏi Thường gặp](#10-câu-hỏi-thường-gặp)

---

# 1. TỔNG QUAN DỰ ÁN

## 1.1 Mục tiêu
Xây dựng hệ thống **nhận diện biển báo giao thông** sử dụng:
- **Mạng nơ-ron nhân tạo (Neural Network)** với PyTorch
- **GPU acceleration** (NVIDIA CUDA)
- **HOG features** cho trích xuất đặc trưng

## 1.2 Dataset sử dụng
**GTSRB (German Traffic Sign Recognition Benchmark):**
- 🖼️ **39,209** ảnh training
- 🖼️ **12,630** ảnh testing  
- 📁 **43 classes** (loại biển báo)
- 📐 Kích thước gốc: khác nhau → resize về **32×32 pixels**

## 1.3 Kết quả đạt được
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| 3 Hidden Layers (512, 256, 128) | **96.02%** | 95.98% | ~45s (GPU) |

---

# 2. PIPELINE XỬ LÝ DỮ LIỆU

## 2.1 Sơ đồ Tổng quan

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────┐
│  Ảnh gốc    │───▶│ Tiền xử lý   │───▶│ Trích xuất    │───▶│  Neural    │
│  (RGB)      │    │ (Preprocess) │    │ HOG Features  │    │  Network   │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────┘
                                                                   │
                                                                   ▼
                                                            ┌────────────┐
                                                            │ Prediction │
                                                            │ (43 class) │
                                                            └────────────┘
```

## 2.2 Tiền xử lý Ảnh (Preprocessing)

```python
def load_image(self, img_path: str) -> np.ndarray:
    # 1. Đọc ảnh bằng OpenCV
    img = cv2.imread(str(img_path))
    
    # 2. Chuyển BGR → RGB (OpenCV mặc định đọc BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize về kích thước chuẩn 32×32
    img = cv2.resize(img, (32, 32))
    
    # 4. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 5. Chuẩn hóa giá trị pixel về [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    return gray
```

### Giải thích cho người hỏi:
> "Ảnh gốc có nhiều kích thước khác nhau, nên đầu tiên phải **resize** về cùng kích thước 32×32 để model có thể xử lý. Sau đó chuyển sang **ảnh xám** vì hình dạng biển báo quan trọng hơn màu sắc. Cuối cùng **chia cho 255** để đưa giá trị về khoảng [0,1], giúp model học ổn định hơn."

---

# 3. TRÍCH XUẤT ĐẶC TRƯNG HOG

## 3.1 HOG là gì?
**HOG (Histogram of Oriented Gradients)** là kỹ thuật trích xuất đặc trưng dựa trên **hướng của gradient** trong ảnh.

### Ý tưởng cốt lõi:
> Hình dạng của đối tượng được mô tả bằng **phân bố hướng** của các cạnh (edges).

## 3.2 Các bước tính HOG

### Bước 1: Tính Gradient
Gradient đo **sự thay đổi cường độ** pixel theo chiều ngang và dọc.

**Công thức:**
$$G_x = I(x+1, y) - I(x-1, y)$$
$$G_y = I(x, y+1) - I(x, y-1)$$

**Độ lớn (Magnitude):**
$$|G| = \sqrt{G_x^2 + G_y^2}$$

**Hướng (Orientation):**
$$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

### Ví dụ trực quan:
```
    Ảnh gốc          Gradient theo X       Gradient theo Y
    ┌───┬───┬───┐    ┌───┬───┬───┐        ┌───┬───┬───┐
    │ 50│100│150│    │   │+100│   │       │   │ +50│   │
    ├───┼───┼───┤    ├───┼───┼───┤        ├───┼───┼───┤
    │ 50│100│150│ →  │   │+100│   │       │   │  0 │   │
    ├───┼───┼───┤    ├───┼───┼───┤        ├───┼───┼───┤
    │100│150│200│    │   │+100│   │       │   │ +50│   │
    └───┴───┴───┘    └───┴───┴───┘        └───┴───┴───┘
    
    Gx(1,1) = 150 - 50 = 100 (cạnh dọc mạnh)
    Gy(1,1) = 100 - 100 = 0   (không có cạnh ngang)
```

### Bước 2: Chia ảnh thành Cells
Ảnh 32×32 được chia thành các **cells** 8×8 pixels:
```
    ┌───┬───┬───┬───┐
    │C1 │C2 │C3 │C4 │   Mỗi cell = 8×8 pixels
    ├───┼───┼───┼───┤   Tổng cộng: 4×4 = 16 cells
    │C5 │C6 │C7 │C8 │
    ├───┼───┼───┼───┤
    │C9 │C10│C11│C12│
    ├───┼───┼───┼───┤
    │C13│C14│C15│C16│
    └───┴───┴───┴───┘
```

### Bước 3: Tạo Histogram cho mỗi Cell
Với mỗi cell, tạo **histogram 9 bins** (0° đến 180°):
```
    Hướng 0°-20°   20°-40°   40°-60° ... 160°-180°
         ↓         ↓         ↓           ↓
       ┌───┐     ┌───┐     ┌───┐       ┌───┐
       │2.5│     │1.2│     │0.8│  ...  │1.5│  ← Vote count
       └───┘     └───┘     └───┘       └───┘
         bin 0    bin 1    bin 2        bin 8
```

### Bước 4: Block Normalization
Gom các cells thành **blocks** 2×2 và **chuẩn hóa** để chống lại thay đổi ánh sáng.

**Công thức L2-Hys Normalization:**
$$v' = \frac{v}{\sqrt{\|v\|_2^2 + \epsilon}}$$

## 3.3 Code HOG trong Project

```python
def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
    features = hog(
        image,
        orientations=9,           # 9 hướng (bins)
        pixels_per_cell=(8, 8),   # Cell 8×8 pixels
        cells_per_block=(2, 2),   # Block 2×2 cells
        block_norm='L2-Hys',      # Chuẩn hóa L2-Hys
        visualize=False,
        feature_vector=True       # Trả về vector 1D
    )
    return features
```

## 3.4 Tính số Features

**Công thức tính số HOG features:**

Với ảnh 32×32:
- Số cells: $\frac{32}{8} \times \frac{32}{8} = 4 \times 4 = 16$ cells
- Số blocks: $(4-1) \times (4-1) = 3 \times 3 = 9$ blocks
- Features per block: $2 \times 2 \times 9 = 36$ features
- **Tổng features: $9 \times 36 = 324$ features**

### Giải thích cho người hỏi:
> "HOG biến một ảnh 32×32 (1024 pixels) thành một vector 324 số. Mỗi số đại diện cho **hướng của các cạnh** trong một vùng nhỏ của ảnh. Ví dụ: biển STOP có nhiều cạnh theo hướng xiên 45° nên các bins tương ứng sẽ có giá trị cao."

---

# 4. KIẾN TRÚC NEURAL NETWORK

## 4.1 Multi-Layer Perceptron (MLP)

MLP là kiến trúc neural network cơ bản gồm các **fully-connected layers**.

```
INPUT (324)     HIDDEN 1 (512)    HIDDEN 2 (256)    HIDDEN 3 (128)    OUTPUT (43)
    ○               ○                 ○                 ○                ○
    ○ ─────────────→ ○ ──────────────→ ○ ──────────────→ ○ ─────────────→ ○
    ○               ○                 ○                 ○                ○
    :               :                 :                 :                :
    ○               ○                 ○                 ○                ○
   324            512               256               128               43
 neurons        neurons           neurons           neurons          neurons
```

## 4.2 Code Kiến trúc

```python
class MLPClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,          # 324 (HOG features)
        hidden_sizes: Tuple,      # (512, 256, 128)
        num_classes: int,         # 43 (loại biển báo)
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_size = input_size    # Bắt đầu từ 324
        
        # Xây dựng hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))  # Linear
            layers.append(nn.BatchNorm1d(hidden_size))        # BatchNorm
            layers.append(nn.ReLU())                          # Activation
            layers.append(nn.Dropout(dropout_rate))           # Dropout
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
```

## 4.3 Kiến trúc Chi tiết (Config tốt nhất: 512-256-128)

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER (324 features)                     │
│                      [HOG vector từ ảnh]                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      HIDDEN LAYER 1                               │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────┐ ┌──────────┐ │
│ │ Linear(324→512) │→│ BatchNorm(512)  │→│  ReLU   │→│Dropout30%│ │
│ │  166,400 params │ │  1,024 params   │ │ (free)  │ │ (free)   │ │
│ └─────────────────┘ └─────────────────┘ └─────────┘ └──────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      HIDDEN LAYER 2                               │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────┐ ┌──────────┐ │
│ │ Linear(512→256) │→│ BatchNorm(256)  │→│  ReLU   │→│Dropout30%│ │
│ │  131,328 params │ │    512 params   │ │ (free)  │ │ (free)   │ │
│ └─────────────────┘ └─────────────────┘ └─────────┘ └──────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      HIDDEN LAYER 3                               │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────┐ ┌──────────┐ │
│ │ Linear(256→128) │→│ BatchNorm(128)  │→│  ReLU   │→│Dropout30%│ │
│ │  32,896 params  │ │    256 params   │ │ (free)  │ │ (free)   │ │
│ └─────────────────┘ └─────────────────┘ └─────────┘ └──────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                 │
│ ┌─────────────────┐                                               │
│ │ Linear(128→43)  │ ────────▶ [Logits cho 43 classes]             │
│ │   5,547 params  │                                               │
│ └─────────────────┘                                               │
└──────────────────────────────────────────────────────────────────┘

TỔNG SỐ PARAMETERS: ~338,000 (có thể train được)
```

---

# 5. CÁC LAYER CHI TIẾT

## 5.1 Linear Layer (Fully Connected / Dense)

### Công thức toán học:
$$y = Wx + b$$

Trong đó:
- $x$ ∈ $\mathbb{R}^{n}$ : Vector đầu vào (n neurons)
- $W$ ∈ $\mathbb{R}^{m \times n}$ : Ma trận trọng số
- $b$ ∈ $\mathbb{R}^{m}$ : Vector bias
- $y$ ∈ $\mathbb{R}^{m}$ : Vector đầu ra (m neurons)

### Ví dụ cụ thể (đơn giản hóa):

```
Input (3 neurons):  x = [0.5, 0.8, 0.2]

Weights (3→2):      W = [[0.1, 0.4, 0.3],
                         [0.2, 0.5, 0.1]]

Bias:               b = [0.1, 0.2]

Tính toán:
y₁ = 0.1×0.5 + 0.4×0.8 + 0.3×0.2 + 0.1 = 0.53
y₂ = 0.2×0.5 + 0.5×0.8 + 0.1×0.2 + 0.2 = 0.72

Output (2 neurons): y = [0.53, 0.72]
```

### Số parameters:
$$\text{params} = m \times n + m = m \times (n + 1)$$

**Ví dụ:** Linear(324 → 512) có: $512 \times 324 + 512 = 166,400$ params

### Giải thích cho người hỏi:
> "Linear layer giống như **tổng có trọng số**. Mỗi neuron output nhận thông tin từ TẤT CẢ neurons input. Weight càng lớn = input đó càng quan trọng. Bias giúp điều chỉnh **ngưỡng kích hoạt** của neuron."

---

## 5.2 Batch Normalization

### Vấn đề giải quyết:
**Internal Covariate Shift** - Phân bố input thay đổi trong quá trình training, làm model học chậm.

### Công thức:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Trong đó:
- $\mu_B$ = mean của mini-batch
- $\sigma_B^2$ = variance của mini-batch  
- $\gamma, \beta$ = learnable parameters
- $\epsilon$ = hằng số nhỏ (tránh chia cho 0)

### Ví dụ:
```
Batch gồm 4 samples, neuron có giá trị:
x = [2.0, 4.0, 6.0, 8.0]

Bước 1: Tính mean và variance
μ = (2+4+6+8)/4 = 5.0
σ² = [(2-5)² + (4-5)² + (6-5)² + (8-5)²]/4 = 5.0

Bước 2: Normalize
x̂ = (x - μ) / √(σ² + ε)
x̂ = [-1.34, -0.45, 0.45, 1.34]  (mean≈0, std≈1)

Bước 3: Scale và Shift (γ=1.5, β=0.5 learned)
y = 1.5 × x̂ + 0.5 = [-1.51, -0.17, 1.17, 2.51]
```

### Lợi ích:
1. **Train nhanh hơn** - Learning rate cao hơn
2. **Ổn định gradient** - Tránh exploding/vanishing gradient
3. **Regularization nhẹ** - Giảm overfitting

### Giải thích cho người hỏi:
> "BatchNorm giống như **chuẩn hóa điểm thi**. Thay vì mỗi lớp có thang điểm khác nhau, ta chuyển tất cả về thang chuẩn (mean=0, std=1). Điều này giúp các layer sau học ổn định hơn."

---

## 5.3 ReLU Activation

### Công thức:
$$f(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Đồ thị:
```
    Output ↑
         3 │            ╱
           │           ╱
         2 │          ╱
           │         ╱
         1 │        ╱
           │       ╱
         0 │──────●────────────→ Input
        -1 │      │
           │      │ (= 0 khi input < 0)
    ───────┴──────┴───────────
          -2  -1  0   1   2   3
```

### Đạo hàm:
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Tại sao cần Activation?
**Không có activation:** Nhiều Linear layers = 1 Linear layer (do tính chất tuyến tính)
$$W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

**Có activation:** Thêm tính **phi tuyến**, cho phép học các pattern phức tạp.

### Ví dụ:
```
Input:  [−2.0, 0.5, −0.3, 1.5]
ReLU:   [ 0.0, 0.5,  0.0, 1.5]
         ↑         ↑
         Các giá trị âm bị "tắt"
```

### Giải thích cho người hỏi:
> "ReLU giống như một **cửa 1 chiều**: chỉ cho tín hiệu dương đi qua, tín hiệu âm bị chặn. Điều này tạo ra **tính phi tuyến** - cho phép network học được các quy luật phức tạp như 'nếu có cạnh tròn VÀ màu đỏ thì là biển cấm'."

---

## 5.4 Dropout

### Ý tưởng:
Trong training, **tắt ngẫu nhiên** một số neurons (với xác suất p).

### Công thức:
$$y_i = \begin{cases} 
\frac{x_i}{1-p} & \text{if neuron được giữ (prob } 1-p\text{)} \\
0 & \text{if neuron bị tắt (prob } p\text{)}
\end{cases}$$

### Minh họa (dropout_rate = 0.3):
```
Training phase:
    Layer trước dropout:    [0.5, 0.8, 0.3, 0.9, 0.2]
    Mask (ngẫu nhiên):      [ 1 ,  0 ,  1 ,  1 ,  0 ]
    Scale (1/(1-0.3)=1.43): [0.71, 0 , 0.43, 1.29, 0]
    
Testing phase:
    Không có dropout, dùng tất cả neurons bình thường
```

### Tại sao Dropout hoạt động?
1. **Ensemble effect**: Mỗi mini-batch có network khác nhau
2. **Chống co-adaptation**: Neurons không thể "dựa dẫm" vào nhau
3. **Robust features**: Buộc mỗi neuron học features độc lập

### Giải thích cho người hỏi:
> "Dropout giống như **học nhóm** nhưng mỗi buổi có vài bạn nghỉ ngẫu nhiên. Điều này buộc mỗi người phải tự học, không dựa dẫm. Kết quả: khi thi (testing), tất cả cùng làm thì sẽ tốt hơn!"

---

## 5.5 Output Layer + Softmax

### Công thức Softmax:
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### Tính chất:
- Output trong khoảng (0, 1)
- Tổng tất cả outputs = 1
- Có thể hiểu như **xác suất**

### Ví dụ chi tiết:
```
Logits từ output layer (43 classes, lấy 3 class đầu):
z = [2.5, 1.2, 0.3, ..., -0.5]

Tính softmax cho 3 class đầu:
e^z = [e^2.5, e^1.2, e^0.3] = [12.18, 3.32, 1.35]
sum(e^z) = 12.18 + 3.32 + 1.35 + ... ≈ 25.6

softmax = [12.18/25.6, 3.32/25.6, 1.35/25.6, ...]
        = [0.476, 0.130, 0.053, ...]
          ↑
          Class 0 có xác suất cao nhất!
          
Prediction = Class 0 (47.6% confidence)
```

### Giải thích cho người hỏi:
> "Softmax chuyển đổi các số thô (logits) thành **xác suất**. Số càng lớn → xác suất càng cao. Tổng luôn = 100%. Ví dụ: model dự đoán 'Speed limit 30' với 95% tự tin."

---

# 6. QUÁ TRÌNH TRAINING

## 6.1 Training Loop Tổng quan

```
FOR mỗi epoch (1 → 50):
    FOR mỗi batch trong training_data:
        
        ┌─────────────────────────────────────┐
        │ 1. FORWARD PROPAGATION              │
        │    Input → Network → Prediction     │
        └─────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────┐
        │ 2. COMPUTE LOSS                     │
        │    Loss = CrossEntropy(pred, label) │
        └─────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────┐
        │ 3. BACKPROPAGATION                  │
        │    Tính gradient: ∂Loss/∂weights    │
        └─────────────────────────────────────┘
                         ↓
        ┌─────────────────────────────────────┐
        │ 4. UPDATE WEIGHTS                   │
        │    w = w - lr × gradient            │
        └─────────────────────────────────────┘
    
    Đánh giá trên validation set
    
    IF val_loss không cải thiện sau 10 epochs:
        Early stopping!
```

## 6.2 Code Training Loop

```python
for epoch in range(num_epochs):
    # === TRAINING PHASE ===
    model.train()  # Bật dropout, cập nhật BatchNorm
    
    for batch_features, batch_labels in train_loader:
        # Chuyển data lên GPU
        batch_features = batch_features.to(device)  # [128, 324]
        batch_labels = batch_labels.to(device)      # [128]
        
        # 1. Forward pass
        optimizer.zero_grad()           # Xóa gradient cũ
        outputs = model(batch_features) # [128, 43] logits
        loss = criterion(outputs, batch_labels)
        
        # 2. Backward pass
        loss.backward()    # Tính gradient
        optimizer.step()   # Cập nhật weights
        
        # Statistics
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_labels).sum().item()
    
    # === VALIDATION PHASE ===
    model.eval()  # Tắt dropout, freeze BatchNorm
    
    with torch.no_grad():  # Không tính gradient
        for batch_features, batch_labels in val_loader:
            # Chỉ forward, không backward
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            # ...
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break  # Dừng training
```

## 6.3 Forward Propagation Chi tiết

Với input $x$ (324 features) qua network (512, 256, 128):

**Layer 1:**
$$h_1 = \text{Dropout}(\text{ReLU}(\text{BN}(W_1 x + b_1)))$$

**Layer 2:**
$$h_2 = \text{Dropout}(\text{ReLU}(\text{BN}(W_2 h_1 + b_2)))$$

**Layer 3:**
$$h_3 = \text{Dropout}(\text{ReLU}(\text{BN}(W_3 h_2 + b_3)))$$

**Output:**
$$z = W_4 h_3 + b_4$$
$$\hat{y} = \text{softmax}(z)$$

### Ví dụ số (đơn giản hóa):
```
Input HOG features: x = [0.5, 0.3, ..., 0.8]  (324 dims)
                         ↓
Layer 1 (324→512):  h1 = ReLU(BN(Wx+b)) = [0.9, 0, 0.7, ..., 0.2]
                         ↓  (Dropout tắt 30% neurons)
Layer 2 (512→256):  h2 = ReLU(BN(Wh1+b)) = [0.4, 0.6, ..., 0.1]
                         ↓
Layer 3 (256→128):  h3 = ReLU(BN(Wh2+b)) = [0.8, 0.3, ..., 0.5]
                         ↓
Output (128→43):    z = Wh3+b = [2.1, 0.5, ..., -1.2]  (logits)
                         ↓
Softmax:            ŷ = [0.62, 0.08, ..., 0.01]  (xác suất)
                         ↓
Prediction:         Class 0 (62% confidence)
```

---

## 6.4 Cross-Entropy Loss

### Công thức:
$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Với **one-hot encoding** (chỉ 1 class đúng):
$$L = -\log(\hat{y}_{\text{true class}})$$

### Ví dụ:
```
True label: Class 14 (STOP sign)
One-hot:    y = [0, 0, ..., 1, ..., 0]  (vị trí 14 = 1)
                              ↑

Prediction: ŷ = [0.01, 0.02, ..., 0.85, ..., 0.01]
                                   ↑
                            ŷ[14] = 0.85

Loss = -log(0.85) = 0.163

Nếu prediction sai (ŷ[14] = 0.05):
Loss = -log(0.05) = 2.996  ← Loss CAO = Sai nhiều!
```

### Đồ thị Loss:
```
    Loss ↑
       5 │  ╲
         │   ╲
       3 │    ╲
         │     ╲
       1 │      ╲───────────
         │                   ╲
       0 └──────────────────────→ ŷ (xác suất đúng)
         0    0.2   0.5   0.8   1.0
         
    → ŷ càng gần 1 (dự đoán đúng) → Loss càng nhỏ
```

### Giải thích cho người hỏi:
> "Loss đo **mức độ sai** của prediction. Nếu model dự đoán đúng với confidence cao (95%), loss rất nhỏ. Nếu dự đoán sai hoặc confidence thấp (5%), loss rất lớn. Mục tiêu training: **giảm loss**."

---

## 6.5 Backpropagation

### Ý tưởng:
Tính **đạo hàm của Loss theo từng weight** bằng **Chain Rule**.

### Chain Rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Minh họa (đơn giản):
```
Forward:    x ──[w]──▶ z = wx ──[ReLU]──▶ y ──[Loss]──▶ L

Backward:   ∂L/∂w ◀── ∂L/∂z ◀── ∂L/∂y ◀── ∂L/∂L = 1
            
            ∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
                  = ∂L/∂y × (1 if z>0 else 0) × x
```

### Trong PyTorch:
```python
loss.backward()  # Tự động tính gradient cho TẤT CẢ weights

# Gradient được lưu trong:
# model.layer1.weight.grad
# model.layer1.bias.grad
# ...
```

### Giải thích cho người hỏi:
> "Backpropagation trả lời câu hỏi: '**Nếu thay đổi weight này một chút, Loss sẽ thay đổi bao nhiêu?**' Nếu gradient dương → giảm weight. Nếu gradient âm → tăng weight. Mục tiêu: tìm weights cho Loss nhỏ nhất."

---

## 6.6 Gradient Descent & Adam Optimizer

### Gradient Descent cơ bản:
$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}$$

Trong đó:
- $\eta$ = learning rate (tốc độ học)
- $\frac{\partial L}{\partial w}$ = gradient

### Adam Optimizer (Sử dụng trong code):
Kết hợp **Momentum** và **Adaptive Learning Rate**.

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    weight_decay=1e-4   # L2 regularization
)
```

**Công thức Adam:**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Trong đó:
- $g_t$ = gradient tại bước t
- $m_t$ = first moment (momentum)
- $v_t$ = second moment (adaptive)
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ (defaults)

### Giải thích cho người hỏi:
> "Adam giống như **leo núi thông minh**. Thay vì đi cùng tốc độ mọi lúc, nó:
> 1. Nhớ hướng đã đi (momentum) - tránh dao động
> 2. Điều chỉnh bước chân (adaptive) - bước lớn ở vùng bằng phẳng, bước nhỏ ở vùng dốc"

---

# 7. CÁC KỸ THUẬT TỐI ƯU

## 7.1 Early Stopping

### Vấn đề: Overfitting
```
    Loss ↑
         │    Training loss ↘↘↘↘↘↘↘↘↘
         │    
         │    Validation loss ↘↘↘↗↗↗↗
         │                      ↑
         │               Điểm dừng tối ưu
         └──────────────────────────────→ Epochs
```

### Code:
```python
patience = 10  # Số epochs chờ đợi
patience_counter = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

# Khôi phục model tốt nhất
model.load_state_dict(best_model_state)
```

### Giải thích:
> "Early stopping giống như **biết lúc nào nên dừng học**. Nếu điểm thi thử (validation) không cải thiện sau 10 lần học thêm, có thể đã đạt giới hạn hoặc đang học thuộc lòng (overfitting)."

---

## 7.2 Learning Rate Scheduler

### Vấn đề:
- LR quá cao → Dao động, không hội tụ
- LR quá thấp → Học quá chậm

### Giải pháp: ReduceLROnPlateau
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Giảm LR khi metric không giảm
    factor=0.5,      # LR mới = LR cũ × 0.5
    patience=5       # Chờ 5 epochs
)

# Sau mỗi epoch
scheduler.step(val_loss)
```

### Minh họa:
```
Epoch   Val Loss   LR
1       0.85       0.001
5       0.52       0.001
10      0.48       0.001
15      0.47       0.001  ← Không giảm 5 epochs
16      0.46       0.0005 ← LR giảm 50%
20      0.44       0.0005
25      0.44       0.0005 ← Không giảm
26      0.43       0.00025 ← LR giảm tiếp
```

---

## 7.3 Weight Initialization

### Xavier Initialization:
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

### Tại sao quan trọng?
- **Quá nhỏ:** Tín hiệu yếu dần qua các layers
- **Quá lớn:** Tín hiệu bùng nổ, gradient không ổn định
- **Xavier:** Giữ variance ổn định qua các layers

---

## 7.4 Weight Decay (L2 Regularization)

### Công thức:
$$L_{\text{total}} = L_{\text{CE}} + \lambda \sum_i w_i^2$$

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # λ = 0.0001
)
```

### Giải thích:
> "Weight decay phạt các weights quá lớn, buộc model đơn giản hóa, tránh overfitting."

---

# 8. CÔNG THỨC TOÁN HỌC TỔNG HỢP

## 8.1 Forward Propagation

$$\mathbf{h}^{(l)} = f\left( \text{BN}\left( \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right) \right)$$

Trong đó:
- $l$ = layer index
- $\mathbf{h}^{(0)} = \mathbf{x}$ (input)
- $f$ = ReLU activation
- BN = Batch Normalization

## 8.2 Loss Function

$$L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{n,c} \log(\hat{y}_{n,c})$$

## 8.3 Gradient (Backpropagation)

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

## 8.4 Weight Update (Adam)

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

---

# 9. GIẢI THÍCH TỪNG PHẦN CODE

## 9.1 Cấu trúc Tổng quan

```
traffic_sign_neural_network_pytorch.py
├── Config                    # Các tham số cấu hình
├── TrafficSignDataset       # PyTorch Dataset class
├── ImageDataLoader          # Load và preprocess ảnh
├── FeatureExtractor         # HOG feature extraction
├── MLPClassifier            # Neural Network model
├── TrainingEngine           # Training loop
├── PyTorchVisualizer        # Visualization
├── print_results_summary    # In kết quả
└── main()                   # Entry point
```

## 9.2 Flow Chart

```
main()
   │
   ├──▶ get_device()           # Kiểm tra GPU
   │
   ├──▶ ImageDataLoader        # Load ảnh từ CSV
   │    └──▶ load_image()      # Preprocess mỗi ảnh
   │
   ├──▶ FeatureExtractor
   │    └──▶ extract_hog_features()  # 32×32 → 324 features
   │
   ├──▶ TrainingEngine
   │    ├──▶ train_and_evaluate_all()
   │    │    ├──▶ StandardScaler     # Normalize features
   │    │    ├──▶ train_test_split   # 90% train, 10% val
   │    │    └──▶ DataLoader         # Batch data
   │    │
   │    └──▶ FOR each config (4 configs)
   │         ├──▶ MLPClassifier()    # Tạo model
   │         ├──▶ train_model()      # Training loop
   │         └──▶ evaluate_model()   # Test & metrics
   │
   └──▶ Visualization & Save Results
```

---

# 10. CÂU HỎI THƯỜNG GẶP

## Q1: Tại sao dùng HOG thay vì raw pixels?
> "HOG nén thông tin **hình dạng** từ 1024 pixels xuống 324 features có ý nghĩa hơn. Raw pixels chứa nhiều noise (nhiễu) và thông tin không liên quan (ví dụ: độ sáng tuyệt đối)."

## Q2: Tại sao cần nhiều hidden layers?
> "Mỗi layer học được các patterns **trừu tượng hơn**:
> - Layer 1: Edges, corners (cạnh, góc)
> - Layer 2: Shapes (hình tròn, tam giác)
> - Layer 3: Objects (biển STOP, biển cấm)"

## Q3: Dropout 30% có nghĩa gì?
> "Trong mỗi lần training, 30% neurons bị tắt ngẫu nhiên. Điều này buộc network **không dựa dẫm** vào một vài neurons mạnh, mà phải học distributed representation."

## Q4: Tại sao cần Early Stopping?
> "Nếu train quá lâu, model sẽ **học thuộc** training data (overfitting). Early stopping dừng khi model đã học đủ tốt trên validation set."

## Q5: 96% accuracy nghĩa là gì?
> "Trong 12,630 ảnh test, model dự đoán đúng ~12,125 ảnh. Chỉ sai ~505 ảnh (4%)."

## Q6: Model có thể nhận diện biển báo Việt Nam không?
> "Không trực tiếp được vì model được train trên biển Đức. Tuy nhiên, cùng pipeline có thể train lại với dataset Việt Nam."

## Q7: Tại sao dùng GPU?
> "GPU có hàng nghìn cores nhỏ, rất phù hợp cho phép nhân ma trận song song. Training trên GPU nhanh hơn 10-50 lần so với CPU."

## Q8: Batch size 128 là gì?
> "Mỗi lần cập nhật weights, model xử lý 128 ảnh cùng lúc. Batch nhỏ = noisy gradient. Batch lớn = stable nhưng chậm. 128 là cân bằng tốt."

---

# 📊 TÓM TẮT KIẾN THỨC

| Khái niệm | Giải thích ngắn |
|-----------|-----------------|
| HOG | Trích xuất hướng của edges trong ảnh |
| Linear Layer | Phép biến đổi tuyến tính y = Wx + b |
| BatchNorm | Chuẩn hóa output để train ổn định |
| ReLU | max(0, x) - thêm tính phi tuyến |
| Dropout | Tắt ngẫu nhiên neurons để chống overfitting |
| Softmax | Chuyển logits thành xác suất |
| Cross-Entropy | Đo độ sai lệch prediction vs truth |
| Backpropagation | Tính gradient bằng chain rule |
| Adam | Optimizer thông minh với momentum |
| Early Stopping | Dừng khi val loss không giảm |

---

**Tài liệu này được tạo cho mục đích thuyết trình và giải thích code.**

*Tác giả: ML Project - Traffic Sign Classification*
