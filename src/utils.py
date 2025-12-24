import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class FocalLoss(nn.Module):
    """
    Focal Loss: Giải pháp cho vấn đề mất cân bằng dữ liệu (Class Imbalance) của GTSRB.
    
    Công thức: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Hệ số cân bằng (thường là 1 hoặc 0.25).
        gamma (float): Tham số tập trung (Focusing parameter). Gamma càng cao, 
                       model càng tập trung sửa lỗi cho các mẫu khó (hard examples).
                       Theo tài liệu, gamma=2.0 là tối ưu cho GTSRB.
        reduction (str): 'mean' (mặc định) hoặc 'sum'.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch_size, Num_classes] (Logits chưa qua Softmax)
        # targets: [Batch_size] (Class indices)
        
        # Tính Cross Entropy Loss bình thường (nhưng không reduction để giữ loss từng mẫu)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Tính pt (xác suất dự đoán đúng lớp target)
        pt = torch.exp(-ce_loss)
        
        # Tính Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def save_checkpoint(state, filename="checkpoint.pth", dir_path="checkpoints"):
    """
    Lưu trạng thái model và optimizer.
    Args:
        state (dict): Dictionary chứa state_dict của model, optimizer, epoch...
        filename (str): Tên file.
        dir_path (str): Thư mục lưu trữ.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)
    print(f"=> Đã lưu checkpoint tại: {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Tải checkpoint để resume training hoặc inference.
    """
    print(f"=> Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) # Load về CPU trước cho an toàn
    
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    start_epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)
    
    print(f"=> Đã tải xong! Epoch: {start_epoch}, Best Acc: {best_acc:.2f}%")
    return start_epoch, best_acc

def calculate_accuracy(loader, model, device="cuda"):
    """
    Hàm tính độ chính xác đơn giản trên toàn bộ loader.
    """
    num_correct = 0
    num_samples = 0
    model.eval() # Chuyển sang chế độ đánh giá
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1) # Lấy class có điểm số cao nhất
            
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    
    model.train() # Chuyển lại về chế độ train
    return float(num_correct) / float(num_samples) * 100