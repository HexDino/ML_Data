# 🧠 Mạng Nơ-ron Nhân Tạo (Neural Network)
## Hướng dẫn toàn diện cho người mới bắt đầu

---

## 📋 Mục lục

1. [Giới thiệu về AI và Machine Learning](#1-giới-thiệu-về-ai-và-machine-learning)
2. [Mạng Nơ-ron là gì?](#2-mạng-nơ-ron-là-gì)
3. [Cấu trúc của Mạng Nơ-ron](#3-cấu-trúc-của-mạng-nơ-ron)
4. [Cách Mạng Nơ-ron Hoạt động](#4-cách-mạng-nơ-ron-hoạt-động)
5. [Các Công thức Toán học Cơ bản](#5-các-công-thức-toán-học-cơ-bản)
6. [Quá trình Huấn luyện](#6-quá-trình-huấn-luyện)
7. [Áp dụng cho Nhận diện Biển báo Giao thông](#7-áp-dụng-cho-nhận-diện-biển-báo-giao-thông)
8. [Hướng dẫn Sử dụng Code](#8-hướng-dẫn-sử-dụng-code)
9. [So sánh với các Phương pháp ML khác](#9-so-sánh-với-các-phương-pháp-ml-khác)

---

## 1. Giới thiệu về AI và Machine Learning

### 🤖 Trí tuệ Nhân tạo (AI) là gì?

**Trí tuệ Nhân tạo (Artificial Intelligence - AI)** là lĩnh vực khoa học máy tính nhằm tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh của con người.

```
                    ╔═══════════════════════════════════╗
                    ║      TRÍ TUỆ NHÂN TẠO (AI)        ║
                    ║  Máy móc có khả năng "suy nghĩ"   ║
                    ╚═══════════════════════════════════╝
                                    │
                    ╔═══════════════════════════════════╗
                    ║    MACHINE LEARNING (ML)          ║
                    ║   Máy học từ dữ liệu              ║
                    ╚═══════════════════════════════════╝
                                    │
                    ╔═══════════════════════════════════╗
                    ║      DEEP LEARNING (DL)           ║
                    ║   Mạng nơ-ron nhiều lớp           ║
                    ╚═══════════════════════════════════╝
```

### 📚 Machine Learning là gì?

**Machine Learning (Học máy)** là một nhánh của AI, trong đó máy tính "học" từ dữ liệu mà không cần được lập trình cụ thể cho từng tác vụ.

**Ví dụ đơn giản:**
- 🐱 Dạy máy tính nhận diện mèo bằng cách cho nó xem hàng ngàn ảnh mèo
- 📧 Lọc email spam bằng cách học từ các email đã được đánh dấu spam/không spam
- 🚦 Nhận diện biển báo giao thông từ hình ảnh

---

## 2. Mạng Nơ-ron là gì?

### 🧬 Lấy cảm hứng từ Não bộ Con người

Mạng nơ-ron nhân tạo được lấy cảm hứng từ cách não bộ con người hoạt động:

```
    NÃO NGƯỜI                          MẠNG NƠ-RON NHÂN TẠO
    ─────────                          ────────────────────
    
    ◉ Neuron sinh học                  ◉ Neuron nhân tạo
        │                                  │
        ├── Nhận tín hiệu              ├── Nhận input
        ├── Xử lý thông tin            ├── Tính toán
        └── Truyền tín hiệu            └── Xuất output
        
    ◉ Synapse (khớp thần kinh)         ◉ Weights (trọng số)
        │                                  │
        └── Kết nối giữa các neuron    └── Kết nối giữa các node
```

### 🔍 Định nghĩa Mạng Nơ-ron

**Mạng Nơ-ron Nhân tạo (Artificial Neural Network - ANN)** là một mô hình tính toán gồm nhiều đơn vị xử lý đơn giản (neurons) được kết nối với nhau, có khả năng học từ dữ liệu và đưa ra dự đoán.

**Ví dụ trực quan:**

```
Hãy tưởng tượng bạn đang dạy một đứa trẻ nhận biết biển báo STOP:

1. Lần 1: Cho xem biển STOP → "Đây là biển STOP"
2. Lần 2: Cho xem biển khác → "Đây không phải biển STOP"
3. Lần 3: Cho xem biển STOP (góc khác) → "Đây là biển STOP"
... (lặp lại nhiều lần)

Sau nhiều lần, đứa trẻ học được các đặc điểm:
- Hình bát giác
- Màu đỏ
- Chữ "STOP" trắng

→ Mạng nơ-ron học theo cách tương tự!
```

---

## 3. Cấu trúc của Mạng Nơ-ron

### 📐 Kiến trúc Cơ bản

Một mạng nơ-ron bao gồm các thành phần:

```
        INPUT LAYER          HIDDEN LAYERS          OUTPUT LAYER
        (Lớp đầu vào)        (Các lớp ẩn)          (Lớp đầu ra)
        
            ○                    ○                      
           ╱│╲                  ╱│╲                     ○ → Class 0
          ╱ │ ╲                ╱ │ ╲                   ╱│╲
         ○  │  ○──────────────○  │  ○────────────────○ │ ○ → Class 1
        ╱│╲ │ ╱│╲            ╱│╲ │ ╱│╲              ╲│╱ │ ╲│╱
       ○ │ ○│○ │ ○──────────○ │ ○│○ │ ○──────────────○  │  ○ → Class 2
        ╲│╱ │ ╲│╱            ╲│╱ │ ╲│╱                ╲ │ ╱
         ○  │  ○──────────────○  │  ○────────────────  ○   → ...
          ╲ │ ╱                ╲ │ ╱                   
           ╲│╱                  ╲│╱                     ○ → Class 42
            ○                    ○
            
       [Features]           [Processing]            [Predictions]
       
       Ví dụ:               Layer 1: 256 neurons    43 loại biển báo
       - HOG features       Layer 2: 128 neurons
       - 324 features       Layer 3: 64 neurons
```

### 🔢 Các Lớp (Layers)

#### 1️⃣ Input Layer (Lớp đầu vào)
- Nhận dữ liệu đầu vào
- Mỗi neuron tương ứng với một feature
- Số neuron = Số features của dữ liệu

```python
# Ví dụ: Với ảnh 32x32 dùng HOG features
# HOG tạo ra 324 features → Input layer có 324 neurons
input_size = 324
```

#### 2️⃣ Hidden Layers (Các lớp ẩn)
- Nơi "sức mạnh" của mạng nơ-ron thể hiện
- Học các patterns phức tạp
- Có thể có 1 hoặc nhiều lớp ẩn

```python
# Ví dụ các kiến trúc:
hidden_layer_1 = (128,)           # 1 lớp ẩn, 128 neurons
hidden_layer_2 = (256, 128)       # 2 lớp ẩn, 256 và 128 neurons
hidden_layer_3 = (512, 256, 128)  # 3 lớp ẩn
```

#### 3️⃣ Output Layer (Lớp đầu ra)
- Đưa ra kết quả dự đoán
- Số neuron = Số classes cần phân loại

```python
# Với 43 loại biển báo
output_size = 43  # 43 neurons, mỗi neuron cho 1 loại
```

### 🔗 Các Thành phần của một Neuron

```
                        ┌─────────────────────────────────────┐
                        │           NEURON                     │
    x₁ ───(w₁)──────>  │                                       │
                        │    z = Σ(wᵢ × xᵢ) + b               │
    x₂ ───(w₂)──────>  │                                       │────> output (a)
                        │    a = activation(z)                 │
    x₃ ───(w₃)──────>  │                                       │
           ...          │                                       │
    xₙ ───(wₙ)──────>  │           + bias (b)                 │
                        └─────────────────────────────────────┘
    
    Trong đó:
    - xᵢ : Inputs (đầu vào)
    - wᵢ : Weights (trọng số) - Được học trong quá trình training
    - b  : Bias (độ lệch) - Được học trong quá trình training
    - z  : Weighted sum (tổng có trọng số)
    - a  : Activation output (đầu ra sau kích hoạt)
```

---

## 4. Cách Mạng Nơ-ron Hoạt động

### 🔄 Quy trình Xử lý (Forward Propagation)

```
BƯỚC 1: NHẬN DỮ LIỆU ĐẦU VÀO
─────────────────────────────
    Ảnh biển báo → Tiền xử lý → Feature Extraction → [x₁, x₂, ..., xₙ]
    
BƯỚC 2: TRUYỀN QUA CÁC LỚP ẨN
─────────────────────────────
    Input → [Lớp 1] → [Lớp 2] → ... → [Lớp n]
           (Linear + Activation) lặp lại mỗi lớp
           
BƯỚC 3: ĐƯA RA DỰ ĐOÁN
─────────────────────────────
    [Lớp cuối] → Softmax → [p₀, p₁, ..., p₄₂]
                           (Xác suất cho mỗi class)
                           
    Dự đoán = Class có xác suất cao nhất
```

### 📊 Ví dụ Trực quan

```
Giả sử ta có ảnh biển báo "STOP" (class 14):

1. Feature Extraction:
   Ảnh → HOG → [0.23, 0.45, 0.12, ..., 0.67]  (324 features)
   
2. Forward Propagation:
   
   Input (324)  →  Hidden 1 (256)  →  Hidden 2 (128)  →  Output (43)
   [0.23, ...]     [0.89, 0.12, ...]   [0.45, ...]        [0.01, 0.02, ..., 0.95, ...]
                                                           ↑        ↑
                                                         class 0  class 14
                                                         
3. Prediction:
   Softmax → [0.01, 0.02, 0.00, ..., 0.95, ..., 0.01]
                                      ↑
                               Class 14 = 95% → Dự đoán: STOP!
```

---

## 5. Các Công thức Toán học Cơ bản

### 📐 5.1. Tính Toán trong Neuron

#### Weighted Sum (Tổng có trọng số):

$$z = \sum_{i=1}^{n} w_i \cdot x_i + b = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

Hoặc viết dưới dạng vector:

$$z = \mathbf{w}^T \cdot \mathbf{x} + b$$

**Ý nghĩa:**
- Mỗi input $x_i$ được "cân nhắc" bởi weight $w_i$
- Weight lớn → Input đó quan trọng
- Weight nhỏ → Input đó ít quan trọng
- Bias $b$ giúp điều chỉnh ngưỡng kích hoạt

### 📐 5.2. Hàm Kích Hoạt (Activation Functions)

#### ReLU (Rectified Linear Unit) - Phổ biến nhất:

$$f(z) = \max(0, z) = \begin{cases} z & \text{nếu } z > 0 \\ 0 & \text{nếu } z \leq 0 \end{cases}$$

```
    Output ↑
         3 │        ╱
           │       ╱
         2 │      ╱
           │     ╱
         1 │    ╱
           │   ╱
         0 │──●──────────────→ Input
        -1 │  │
           │  │
    ───────┴──┴───────────────
          -2 -1  0  1  2  3
```

**Ưu điểm:** Đơn giản, tính toán nhanh, tránh vanishing gradient

---

#### Sigmoid:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
    Output ↑
       1.0 │            ●●●●●
           │          ●
       0.5 │        ●
           │      ●
       0.0 │●●●●●
           └────────────────→ Input
             -5    0    5
```

**Đặc điểm:** Output trong khoảng (0, 1), thường dùng cho binary classification

---

#### Tanh (Hyperbolic Tangent):

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

```
    Output ↑
       1.0 │            ●●●●●
           │          ●
       0.0 │────────●────────→ Input
           │      ●
      -1.0 │●●●●●
           └────────────────
             -5    0    5
```

**Đặc điểm:** Output trong khoảng (-1, 1), zero-centered

---

#### Softmax (cho Output Layer):

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Đặc điểm:**
- Chuyển đổi vector thành xác suất
- Tổng tất cả outputs = 1
- Dùng cho multi-class classification

```python
# Ví dụ Softmax:
z = [2.0, 1.0, 0.1]  # Raw scores từ 3 classes

# Tính softmax:
e_z = [e^2.0, e^1.0, e^0.1] = [7.39, 2.72, 1.11]
sum_e_z = 7.39 + 2.72 + 1.11 = 11.22

softmax = [7.39/11.22, 2.72/11.22, 1.11/11.22]
        = [0.66, 0.24, 0.10]  # Xác suất cho mỗi class
        
# Class 0 có xác suất cao nhất → Dự đoán class 0
```

### 📐 5.3. Loss Function (Hàm Mất mát)

#### Cross-Entropy Loss (cho Classification):

$$L = -\sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)$$

Trong đó:
- $C$ = số classes
- $y_i$ = label thực (0 hoặc 1)
- $\hat{y}_i$ = xác suất dự đoán

**Ví dụ:**
```
Thực tế: Class 2 (y = [0, 0, 1])
Dự đoán: [0.1, 0.2, 0.7]

Loss = -[0×log(0.1) + 0×log(0.2) + 1×log(0.7)]
     = -log(0.7)
     = 0.357

→ Loss càng nhỏ = Dự đoán càng đúng
```

---

## 6. Quá trình Huấn luyện

### 🔄 Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PROCESS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  STEP 1: FORWARD PROPAGATION                          │     │
│    │  ───────────────────────────────                      │     │
│    │  Input → Hidden Layers → Output → Prediction (ŷ)      │     │
│    └──────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  STEP 2: COMPUTE LOSS                                 │     │
│    │  ──────────────────                                   │     │
│    │  Loss = CrossEntropy(y_true, y_pred)                  │     │
│    │  So sánh dự đoán với thực tế                          │     │
│    └──────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  STEP 3: BACKPROPAGATION                              │     │
│    │  ─────────────────────                                │     │
│    │  Tính gradient của loss theo mỗi weight               │     │
│    │  ∂L/∂w cho tất cả weights                             │     │
│    └──────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  STEP 4: UPDATE WEIGHTS                               │     │
│    │  ────────────────────                                 │     │
│    │  w_new = w_old - learning_rate × gradient             │     │
│    │  Điều chỉnh weights để giảm loss                      │     │
│    └──────────────────────────────────────────────────────┘     │
│                            ↓                                     │
│                    Lặp lại cho đến khi:                          │
│                    - Loss đủ nhỏ                                 │
│                    - Đạt số iterations tối đa                    │
│                    - Early stopping                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 📉 Gradient Descent - Thuật toán Tối ưu

**Công thức cập nhật weight:**

$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$

Trong đó:
- $\eta$ = learning rate (tốc độ học)
- $\frac{\partial L}{\partial w}$ = gradient của loss theo weight

**Ví dụ trực quan:**

```
                    Loss
                      ↑
                      │    ╲
                      │     ╲    ← Bắt đầu ở đây
                      │      ╲
                      │       ○  ← Bước 1
                      │        ╲
                      │         ○ ← Bước 2
                      │          ╲
                      │           ○ ← Bước 3
                      │            ●  ← Đích (minimum)
                      └────────────────────→ Weight
                      
    Mục tiêu: "Leo xuống" để tìm điểm có Loss nhỏ nhất
```

### ⚙️ Các Optimizer phổ biến

#### 1. SGD (Stochastic Gradient Descent)
```python
w = w - learning_rate * gradient
```

#### 2. Adam (Adaptive Moment Estimation) - **Khuyên dùng**
- Kết hợp momentum và adaptive learning rate
- Tự động điều chỉnh learning rate cho mỗi parameter

```python
# Adam update (simplified):
m = β₁ * m + (1 - β₁) * gradient        # First moment
v = β₂ * v + (1 - β₂) * gradient²       # Second moment
w = w - learning_rate * m / (√v + ε)
```

---

## 7. Áp dụng cho Nhận diện Biển báo Giao thông

### 🚦 Pipeline Xử lý

```
┌────────────────────────────────────────────────────────────────────┐
│                    TRAFFIC SIGN RECOGNITION                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INPUT IMAGE                    2. PREPROCESSING                 │
│  ┌─────────────┐                   ┌─────────────────────────────┐ │
│  │  🚦         │   ──────────>     │  • Resize to 32x32          │ │
│  │  [STOP]     │                   │  • Convert to Grayscale     │ │
│  │             │                   │  • Normalize to [0, 1]      │ │
│  └─────────────┘                   └─────────────────────────────┘ │
│                                                                     │
│  3. FEATURE EXTRACTION             4. NEURAL NETWORK               │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐ │
│  │  HOG (Histogram of          │  │  Input: 324 features        │ │
│  │  Oriented Gradients)        │  │  Hidden: (256, 128)         │ │
│  │                             │  │  Output: 43 classes         │ │
│  │  → 324 features             │  │  Activation: ReLU           │ │
│  └─────────────────────────────┘  └─────────────────────────────┘ │
│                                                                     │
│  5. OUTPUT                                                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Prediction: Class 14 (STOP sign) - Confidence: 95.2%       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 📊 Dataset GTSRB

**German Traffic Sign Recognition Benchmark:**
- 🖼️ 39,209 ảnh training
- 🖼️ 12,630 ảnh testing
- 📁 43 loại biển báo
- 📐 Kích thước khác nhau → Resize về 32x32

### 🏗️ Các Kiến trúc Được Test

| Kiến trúc | Hidden Layers | Số Parameters (approx) |
|-----------|---------------|------------------------|
| NN_1Layer | (128) | ~42,000 |
| NN_2Layer | (256, 128) | ~120,000 |
| NN_3Layer | (512, 256, 128) | ~270,000 |
| NN_3Layer_v2 | (256, 128, 64) | ~85,000 |

---

## 8. Hướng dẫn Sử dụng Code

### 🚀 Quick Start

```bash
# 1. Chạy đầy đủ với tất cả configurations
python traffic_sign_neural_network.py

# 2. Chạy nhanh (test với ít dữ liệu)
python traffic_sign_neural_network.py --quick

# 3. Chạy với sample size tùy chỉnh
python traffic_sign_neural_network.py --quick --sample-size 10000
```

### 📝 Các Options có sẵn

```bash
# Xem tất cả options
python traffic_sign_neural_network.py --help

Options:
  --quick           Chạy nhanh với subset dữ liệu
  --sample-size N   Số samples cho quick mode (default: 5000)
  --use-folders     Load data từ folders thay vì CSV
  --split-train     Chia train data thay vì dùng test set riêng
  --features METHOD Feature extraction: 'hog' hoặc 'pixel'
  --custom          Train một neural network tùy chỉnh
  --layers SIZES    Kích thước hidden layers (e.g., "256,128,64")
```

### 🔧 Custom Training

```bash
# Train với kiến trúc tùy chỉnh
python traffic_sign_neural_network.py --custom --layers "512,256,128"

# Train với raw pixels thay vì HOG
python traffic_sign_neural_network.py --features pixel
```

### 📊 Output Files

Sau khi chạy, chương trình sẽ tạo ra các files:

| File | Mô tả |
|------|-------|
| `nn_architecture_*.png` | Hình minh họa kiến trúc mạng |
| `nn_training_curves.png` | Đồ thị loss trong quá trình training |
| `nn_model_comparison.png` | So sánh các configurations |
| `nn_confidence_distribution.png` | Phân bố độ tin cậy |
| `nn_confusion_matrix_*.png` | Ma trận nhầm lẫn |
| `nn_results_summary.csv` | Kết quả dạng bảng |

---

## 9. So sánh với các Phương pháp ML khác

### 📊 Bảng So sánh

| Tiêu chí | SVM | KNN | Random Forest | Neural Network |
|----------|-----|-----|---------------|----------------|
| **Độ phức tạp** | Trung bình | Thấp | Trung bình | Cao |
| **Thời gian train** | Lâu | Nhanh | Trung bình | Lâu |
| **Thời gian inference** | Nhanh | Chậm | Nhanh | Nhanh |
| **Khả năng mở rộng** | Kém | Kém | Tốt | Rất tốt |
| **Xử lý dữ liệu phức tạp** | Trung bình | Kém | Tốt | Rất tốt |
| **Yêu cầu dữ liệu** | Ít | Ít | Trung bình | Nhiều |
| **Điều chỉnh hyperparameters** | Trung bình | Ít | Nhiều | Nhiều |

### 🎯 Khi nào nên dùng Neural Network?

✅ **Nên dùng khi:**
- Có nhiều dữ liệu training (>10,000 samples)
- Bài toán phức tạp, nhiều classes
- Cần độ chính xác cao
- Có thời gian và tài nguyên training

❌ **Không nên dùng khi:**
- Dữ liệu ít (<1,000 samples)
- Cần giải thích được model (interpretability)
- Tài nguyên tính toán hạn chế
- Bài toán đơn giản

---

## 📚 Thuật ngữ Quan trọng

| Thuật ngữ | Tiếng Việt | Ý nghĩa |
|-----------|------------|---------|
| Neuron | Nơ-ron | Đơn vị tính toán cơ bản |
| Weight | Trọng số | Tham số kết nối giữa neurons |
| Bias | Độ lệch | Tham số điều chỉnh ngưỡng |
| Activation | Hàm kích hoạt | Thêm tính phi tuyến |
| Forward Propagation | Lan truyền tiến | Tính output từ input |
| Backpropagation | Lan truyền ngược | Tính gradient để cập nhật |
| Loss | Mất mát | Độ sai lệch giữa dự đoán và thực tế |
| Gradient | Gradient | Đạo hàm của loss theo weight |
| Learning Rate | Tốc độ học | Bước nhảy khi cập nhật weight |
| Epoch | Epoch | Một lần duyệt qua toàn bộ data |
| Batch | Batch | Nhóm samples được xử lý cùng lúc |
| Overfitting | Quá khớp | Model học thuộc training data |
| Regularization | Điều chuẩn | Kỹ thuật tránh overfitting |

---

## 🎓 Tài liệu Tham khảo

1. **Deep Learning Book** - Ian Goodfellow et al.
2. **Neural Networks and Deep Learning** - Michael Nielsen (free online)
3. **Scikit-learn Documentation** - MLPClassifier
4. **GTSRB Dataset** - German Traffic Sign Recognition Benchmark

---

## 📝 Tác giả

**ML Project - Traffic Sign Classification**

*University Project: Comparing Traditional ML vs Neural Networks*

---

> 💡 **Tip:** Bắt đầu với `--quick` mode để hiểu cách hoạt động, sau đó chạy full training để có kết quả chính xác hơn!
