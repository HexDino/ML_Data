import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    BasicBlock: Khối cơ bản của ResNet-18 và ResNet-34.
    Gồm 2 lớp Conv 3x3 liên tiếp và một đường tắt (Skip Connection).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 1. Conv Layer thứ nhất
        # Nếu stride=2 thì ảnh sẽ giảm kích thước ở lớp này
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2. Conv Layer thứ hai
        # Luôn luôn là stride=1 để giữ nguyên kích thước sau conv1
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 3. Shortcut (Đường tắt - Skip Connection)
        # Nếu kích thước ảnh thay đổi (stride > 1) hoặc số channels thay đổi,
        # ta cần một lớp Conv 1x1 trên đường tắt để khớp dimensions cộng lại được.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # Lưu lại input ban đầu (identity)
        residual = x

        # Đi qua nhánh chính (Weight Layer)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Cộng với nhánh tắt (Shortcut)
        # Đây là mấu chốt giải quyết Vanishing Gradient
        out += self.shortcut(residual)
        
        # Activation cuối cùng
        out = F.relu(out)
        return out

class ResNet18_Custom(nn.Module):
    """
    Kiến trúc ResNet-18 Tùy chỉnh (Customized) cho GTSRB.
    Tham khảo: Mục 6.2 trong Báo cáo chuyên sâu.
    """
    def __init__(self, num_classes=43):
        super(ResNet18_Custom, self).__init__()
        self.in_channels = 64

        # --- PHẦN 1: CUSTOM STEM (ĐẦU MẠNG) ---
        # ResNet gốc: Conv 7x7 stride 2 -> MaxPool 3x3 stride 2 (Giảm size 4 lần).
        # Custom GTSRB: Conv 3x3 stride 1 (Giữ nguyên size).
        # Lý do: Ảnh 48x48 quá nhỏ, nếu giảm size sớm sẽ mất hết chi tiết[cite: 122].
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Lưu ý: Đã loại bỏ hoàn toàn lớp MaxPool ở đây theo tài liệu[cite: 123].

        # --- PHẦN 2: CÁC LỚP RESIDUAL (BODY) ---
        # Layer 1: 64 filters, stride 1 (Output: 48x48) [cite: 127]
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)
        
        # Layer 2: 128 filters, stride 2 (Output: 24x24) [cite: 128]
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        
        # Layer 3: 256 filters, stride 2 (Output: 12x12) [cite: 129]
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)
        
        # Layer 4: 512 filters, stride 2 (Output: 6x6) [cite: 130]
        # (Lưu ý: Nếu ảnh input 32x32 thì đến đây còn 4x4, input 48x48 thì còn 6x6)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)

        # --- PHẦN 3: HEAD (ĐUÔI MẠNG) ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling [cite: 132]
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes) # Fully Connected [cite: 134]

        # Khởi tạo trọng số (Weight Initialization) cho tốt
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Hàm tạo một layer gồm nhiều block liên tiếp.
        Block đầu tiên chịu trách nhiệm downsampling (nếu stride=2).
        Các block sau giữ nguyên kích thước.
        """
        strides = [stride] + [1]*(num_blocks-1) # Ví dụ: [2, 1] cho layer stride=2
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Kaiming He Initialization - Tốt cho mạng dùng ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # Không có MaxPool

        # 2. Body
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 3. Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

# --- Sanity Check (Kiểm tra nhanh) ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18_Custom(num_classes=43).to(device)
    
    # Giả lập ảnh đầu vào 48x48
    x = torch.randn(2, 3, 48, 48).to(device)
    y = model(x)
    
    print(f"Model Architecture: ResNet18 Customized")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}") # Kỳ vọng: [2, 43]
    
    # Kiểm tra số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}") 
    # ResNet18 gốc khoảng 11M, bản này cũng tầm đó nhưng hiệu quả hơn cho ảnh nhỏ.