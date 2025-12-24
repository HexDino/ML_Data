import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from resnet import ResNet18_Custom
from utils import FocalLoss, save_checkpoint

# --- CẤU HÌNH ---
CONFIG = {
    "DATA_DIR": "data",
    "CHECKPOINT_DIR": "checkpoints",
    "LOG_DIR": "logs",
    "LOG_FILE": "train_log.csv",
    "MODEL_NAME": "resnet18_custom_best.pth",
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_CLASSES": 43,
    "IMAGE_SIZE": 48,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 8,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "NUM_EPOCHS": 30,
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

def log_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, lr, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'LR'])
        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}", lr])

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, epoch_index):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch_index}", leave=True)
    
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(CONFIG["DEVICE"])
        targets = targets.to(CONFIG["DEVICE"])

        # Forward & Loss
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Tính toán metrics
        total_loss += loss.item()
        
        # Tính Accuracy ngay trong lúc train
        _, preds = predictions.max(1)
        num_correct += (preds == targets).sum().item()
        num_samples += preds.size(0)

        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(loader)
    avg_acc = (num_correct / num_samples) * 100
    return avg_loss, avg_acc

def evaluate(loader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(CONFIG["DEVICE"])
            targets = targets.to(CONFIG["DEVICE"])
            
            scores = model(data)
            loss = loss_fn(scores, targets)
            
            total_loss += loss.item()
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum().item()
            num_samples += predictions.size(0)
    
    avg_loss = total_loss / len(loader)
    avg_acc = (num_correct / num_samples) * 100
    return avg_loss, avg_acc

def main():
    seed_everything(CONFIG["SEED"])
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
    log_path = os.path.join(CONFIG["LOG_DIR"], CONFIG["LOG_FILE"])

    if os.path.exists(log_path):
        os.remove(log_path)

    train_loader, val_loader = get_dataloaders(
        data_dir=CONFIG["DATA_DIR"],
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=CONFIG["IMAGE_SIZE"],
        val_split=CONFIG["VAL_SPLIT"],
        seed=CONFIG["SEED"],
        num_workers=CONFIG["NUM_WORKERS"]
    )

    print("🏗️  Khởi tạo ResNet-18 Custom...")
    model = ResNet18_Custom(num_classes=CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])

    loss_fn = FocalLoss(gamma=CONFIG["GAMMA"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["NUM_EPOCHS"])
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    print("🔥 Bắt đầu Training...")
    for epoch in range(1, CONFIG["NUM_EPOCHS"] + 1):
        
        # Train
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, epoch)
        
        # Val (Truyền thêm loss_fn để tính Val Loss)
        val_loss, val_acc = evaluate(val_loader, model, loss_fn)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        # Ghi log
        log_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, log_path)

        # Lưu model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }, filename=CONFIG["MODEL_NAME"], dir_path=CONFIG["CHECKPOINT_DIR"])
            print(f"   ✅ Đã lưu model tốt nhất: {val_acc:.2f}%")

    print(f"\n🎉 Hoàn tất! Log chi tiết tại: {log_path}")

if __name__ == "__main__":
    main()