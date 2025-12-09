import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import warnings

from model import UNet
from dataset import DIV2KDataset
from utils import TotalLoss, get_color_reduced_image
from utils_diff_jpeg_new import differentiable_jpeg

# --- Parameters ---
base_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_HR_DIR = os.path.join(base_dir, 'data', 'DIV2K_train_HR', 'images')
VALID_HR_DIR = os.path.join(base_dir, 'data', 'DIV2K_valid_HR', 'images')
BATCH_SIZE = 25
PATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_REPRESENTATIVE_COLORS = 4 
GAUSSIAN_KERNEL_SIZE = 1 
JPEG_QUALITY = 50.0 
# ---

def save_debug_images(original, reduced, predicted, jpeg_output, epoch):
    """儲存訓練過程中的預覽圖以便除錯"""
    os.makedirs("debug_images", exist_ok=True)
    # 取 Batch 中的第一張圖，並確保維度正確
    img_list = []
    for img in [original, reduced, predicted, jpeg_output]:
        img = img[0].detach().cpu()
        img = (img + 1) / 2.0 # 反正規化 [-1, 1] -> [0, 1]
        img = torch.clamp(img, 0, 1)
        img_list.append(img)
        
    # 拼接成一張大圖 (Rows: Original, Reduced, Predicted, JPEG)
    comparison = torch.stack(img_list)
    save_image(comparison, f"debug_images/epoch_{epoch+1}.png", nrow=4, padding=2)

def main():
    print(f"Using device: {DEVICE}")
    warnings.filterwarnings("ignore") 

    if not os.path.exists(TRAIN_HR_DIR):
        print(f"Error: {TRAIN_HR_DIR} not found.")
        return

    print("Loading dataset...")
    train_dataset = DIV2KDataset(hr_dir=TRAIN_HR_DIR, patch_size=PATCH_SIZE)
    valid_dataset = DIV2KDataset(hr_dir=VALID_HR_DIR, patch_size=PATCH_SIZE)
    
    num_workers = 4 if os.name != 'nt' else 0 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Dataset loaded. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    model = UNet().to(DEVICE)
    # 建立 Loss 計算模組 (注意：我們會在 loop 中手動加權，所以這裡只用來計算分量)
    criterion = TotalLoss(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    train_history = {'total': [], 'l1': [], 'vgg': [], 'lpips': []}
    
    print("\n--- Training Start ---")
    # print("Strategy: Warmup with only L1 loss for first 10 epochs.")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = {'total': 0, 'l1': 0, 'vgg': 0, 'lpips': 0}
        
        # --- 動態權重調整 (Warmup 策略) ---
        if epoch < 0: # 不做這個
            w_l1, w_vgg, w_lpips = 1.0, 0.0, 0.0 # 前 10 輪只看 L1 (結構)
        else:
            w_l1, w_vgg, w_lpips = 0.8, 0.01, 0.4 # 後期加入感知損失，但權重調低避免干擾 1.0, 0.01, 0.1
        # ----------------------------------

        progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        
        for i, original_images in enumerate(progress_bar):
            original_images = original_images.to(DEVICE)

            gaussian_filtered_images = gaussian_blur(original_images, kernel_size=[GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE])
            
            with torch.no_grad():
                color_reduced_images = get_color_reduced_image(original_images, NUM_REPRESENTATIVE_COLORS)

            color_residuals = model(color_reduced_images, original_images)
            predicted_images = color_reduced_images + color_residuals
            predicted_images = torch.clamp(predicted_images, -1.0, 1.0)

            jpeg_coded_images = differentiable_jpeg(predicted_images, quality=JPEG_QUALITY)

            # 取得各個 Loss 分量 (忽略 utils.py 裡預設的 total)
            _, l1, vgg, lpips_val = criterion(jpeg_coded_images, gaussian_filtered_images)

            # 手動計算加權總 Loss
            loss = (w_l1 * l1) + (w_vgg * vgg) + (w_lpips * lpips_val)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁減
            optimizer.step()

            # 累計
            epoch_loss['total'] += loss.item()
            epoch_loss['l1'] += l1.item()
            epoch_loss['vgg'] += vgg.item()
            epoch_loss['lpips'] += lpips_val.item()
            
            progress_bar.set_postfix({'L': loss.item(), 'L1': l1.item()})

        # 計算平均 Loss
        avg_loss = {k: v / len(train_loader) for k, v in epoch_loss.items()}
        
        # 記錄歷史
        for k in train_history:
            train_history[k].append(avg_loss[k])

        # --- Validation ---
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for i, original_images in enumerate(valid_loader):
                original_images = original_images.to(DEVICE)
                gaussian_filtered_images = gaussian_blur(original_images, kernel_size=[GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE])
                color_reduced_images = get_color_reduced_image(original_images, NUM_REPRESENTATIVE_COLORS)
                
                color_residuals = model(color_reduced_images, original_images)
                predicted_images = torch.clamp(color_reduced_images + color_residuals, -1.0, 1.0)
                jpeg_coded_images = differentiable_jpeg(predicted_images, quality=JPEG_QUALITY)
                
                _, l1, vgg, lpips_val = criterion(jpeg_coded_images, gaussian_filtered_images)
                
                # Validation 使用相同的加權標準
                val_loss = (w_l1 * l1) + (w_vgg * vgg) + (w_lpips * lpips_val)
                val_loss_acc += val_loss.item()
                
                if i == 0:
                    save_debug_images(original_images, color_reduced_images, predicted_images, jpeg_coded_images, epoch)

        avg_val_loss = val_loss_acc / len(valid_loader)
        scheduler.step()

        # 顯示詳細數據
        print(f"Ep {epoch+1} | Total: {avg_loss['total']:.4f} (Val: {avg_val_loss:.4f}) | "
              f"L1: {avg_loss['l1']:.4f} | VGG: {avg_loss['vgg']:.4f} | LPIPS: {avg_loss['lpips']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> Model saved (Best Val: {best_val_loss:.4f})")

    plot_losses(train_history)

def plot_losses(history, filename='loss_components.png'):
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(history['total']) + 1)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['total'], label='Total Loss', color='blue')
    plt.title('Total Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['l1'], label='L1 Loss', color='orange')
    plt.title('L1 Loss (Structure)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['vgg'], label='VGG Loss', color='green')
    plt.title('VGG Loss (Feature)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lpips'], label='LPIPS Loss', color='red')
    plt.title('LPIPS Loss (Perceptual)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Loss components plot saved to '{filename}'")

if __name__ == '__main__':
    main()