import os
import csv
import random
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from resnet import ResNet18_Custom
from utils import FocalLoss, save_checkpoint, load_checkpoint

# --- CẤU HÌNH FINETUNE ---
CONFIG = {
    "DATA_DIR": "data",
    "CHECKPOINT_DIR": "checkpoints",
    "LOG_DIR": "logs",
    "LOG_FILE": "finetune_log.csv",
    "LOAD_MODEL_NAME": "resnet18_custom_best.pth",
    "SAVE_MODEL_NAME": "resnet18_custom_finetuned.pth",
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_CLASSES": 43,
    "IMAGE_SIZE": 48,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 8,
    
    # Finetune settings: LR nhỏ hơn, chạy ít epoch hơn
    "LEARNING_RATE": 1e-4,  # Nhỏ hơn 10 lần so với lúc train (1e-3)
    "WEIGHT_DECAY": 1e-3,   # Tăng nhẹ weight decay để tránh overfitting khi finetune
    "NUM_EPOCHS": 10,       # Chạy thêm 10 epochs
    "VAL_SPLIT": 0.2,
    "GAMMA": 2.0,
    "SEED": 42
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_to_csv(epoch, train_loss, val_acc, lr, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Acc', 'LR'])
        writer.writerow([epoch, train_loss, val_acc, lr])

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch_index):
    model.train()
    loop = tqdm(loader, desc=f"Finetune Epoch {epoch_index}", leave=True)
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(CONFIG["DEVICE"])
        targets = targets.to(CONFIG["DEVICE"])

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(CONFIG["DEVICE"])
            targets = targets.to(CONFIG["DEVICE"])
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
    return float(num_correct) / float(num_samples) * 100

def main():
    seed_everything(CONFIG["SEED"])
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    log_path = os.path.join(CONFIG["LOG_DIR"], CONFIG["LOG_FILE"])

    # Xóa log cũ
    if os.path.exists(log_path):
        os.remove(log_path)

    # 1. Load Data
    print("🔄 Đang chuẩn bị dữ liệu cho Finetune...")
    train_loader, val_loader = get_dataloaders(
        data_dir=CONFIG["DATA_DIR"],
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=CONFIG["IMAGE_SIZE"],
        val_split=CONFIG["VAL_SPLIT"],
        seed=CONFIG["SEED"],
        num_workers=CONFIG["NUM_WORKERS"]
    )

    # 2. Khởi tạo Model
    print("🏗️  Khởi tạo ResNet-18 Custom...")
    model = ResNet18_Custom(num_classes=CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])

    # 3. Load Checkpoint Tốt Nhất Trước Đó
    ckpt_path = os.path.join(CONFIG["CHECKPOINT_DIR"], CONFIG["LOAD_MODEL_NAME"])
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy checkpoint: {ckpt_path}. Vui lòng train trước!")
        return

    print(f"📥 Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    prev_best_acc = checkpoint.get('best_acc', 0.0)
    print(f"   Model trước đó đạt Acc: {prev_best_acc:.2f}%")

    # 4. Setup Training mới (Reset Optimizer & Scheduler)
    # Lý do reset: Finetune cần LR nhỏ và ổn định, không nên dùng tiếp state của Optimizer cũ
    loss_fn = FocalLoss(gamma=CONFIG["GAMMA"])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    
    # Dùng CosineAnnealing cho 10 epoch ngắn ngủi
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["NUM_EPOCHS"])
    
    scaler = torch.cuda.amp.GradScaler()

    best_acc = prev_best_acc # Mốc so sánh là kỷ lục cũ

    print(f"🔥 Bắt đầu Finetune (LR={CONFIG['LEARNING_RATE']})...")
    
    for epoch in range(1, CONFIG["NUM_EPOCHS"] + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch)
        val_acc = evaluate(val_loader, model)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"   Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)")
        
        log_to_csv(epoch, train_loss, val_acc, current_lr, log_path)

        # Lưu nếu tốt hơn (hoặc bằng nhưng loss thấp hơn - tuỳ logic, ở đây lấy Acc cao hơn)
        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename=CONFIG["SAVE_MODEL_NAME"], dir_path=CONFIG["CHECKPOINT_DIR"])
            print(f"   ✅ Đã lưu model finetune mới!")

    print(f"\n🎉 Finetune hoàn tất! Model lưu tại checkpoints/{CONFIG['SAVE_MODEL_NAME']}")

if __name__ == "__main__":
    main()