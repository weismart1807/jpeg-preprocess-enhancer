import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import os
import matplotlib.pyplot as plt 
import warnings # For suppressing warnings

from model import UNet
from dataset import DIV2KDataset
from utils import TotalLoss, get_color_reduced_image

# --- Select Differentiable JPEG Implementation ---
# from utils_diff_jpeg_new import differentiable_jpeg
from utils_diff_jpeg_old import differentiable_jpeg
# from utils_diff_jpeg_simple import differentiable_jpeg
# ---

# --- Parameters ---
base_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_HR_DIR = os.path.join(base_dir, 'data', 'DIV2K_train_HR', 'images')
VALID_HR_DIR = os.path.join(base_dir, 'data', 'DIV2K_valid_HR', 'images')
BATCH_SIZE = 16
PATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
NUM_REPRESENTATIVE_COLORS = 4 # N=4
GAUSSIAN_KERNEL_SIZE = 3 # 3x3
# ---

def main():
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    train_dataset = DIV2KDataset(hr_dir=TRAIN_HR_DIR, patch_size=PATCH_SIZE)
    valid_dataset = DIV2KDataset(hr_dir=VALID_HR_DIR, patch_size=PATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("Dataset loaded.")

    model = UNet().to(DEVICE)
    criterion = TotalLoss(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Learning rate decays to 1/10th every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_loss = float('inf')

    train_losses_history = []
    valid_losses_history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_acc = {'total': 0, 'l1': 0, 'vgg': 0, 'lpips': 0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", leave=False)
        for original_images in progress_bar:
            original_images = original_images.to(DEVICE)

            # 1. Create the target image (Gaussian filtered)
            gaussian_filtered_images = gaussian_blur(original_images, kernel_size=[GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE])
            
            # 2. Create the color-reduced input
            with torch.no_grad(): # This step does not require gradients
                color_reduced_images = get_color_reduced_image(original_images, NUM_REPRESENTATIVE_COLORS)

            # 3. Predict the color residual
            color_residuals = model(color_reduced_images, original_images)
            
            # 4. Generate the predicted image
            predicted_images = color_reduced_images + color_residuals
            predicted_images = torch.clamp(predicted_images, -1.0, 1.0)

            # 5. Apply differentiable JPEG compression
            jpeg_coded_images = differentiable_jpeg(predicted_images)

            # 6. Calculate loss
            loss, l1, vgg, lpips_val = criterion(jpeg_coded_images, gaussian_filtered_images)

            # 7. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            train_loss_acc['total'] += loss.item()
            train_loss_acc['l1'] += l1.item()
            train_loss_acc['vgg'] += vgg.item()
            train_loss_acc['lpips'] += lpips_val.item()

        scheduler.step()

        avg_train_loss = train_loss_acc['total'] / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            progress_bar_val = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", leave=False)
            for original_images in progress_bar_val:
                original_images = original_images.to(DEVICE)
                
                gaussian_filtered_images = gaussian_blur(original_images, kernel_size=[GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE])
                color_reduced_images = get_color_reduced_image(original_images, NUM_REPRESENTATIVE_COLORS)
                
                color_residuals = model(color_reduced_images, original_images)
                predicted_images = color_reduced_images + color_residuals
                predicted_images = torch.clamp(predicted_images, -1.0, 1.0)
                
                jpeg_coded_images = differentiable_jpeg(predicted_images)
                
                loss, _, _, _ = criterion(jpeg_coded_images, gaussian_filtered_images)
                val_loss_acc += loss.item()

        avg_val_loss = val_loss_acc / len(valid_loader)
        # --- End Validation Loop ---

        train_losses_history.append(avg_train_loss)
        valid_losses_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Validation loss improved. Saving model to 'best_model.pth'")

    print("Training complete!")

    # Plot and save the loss curves
    plot_losses(train_losses_history, valid_losses_history)


def plot_losses(train_losses, valid_losses, filename='loss_plot.png'):
    """Plots and saves the training and validation loss curves."""
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o', linestyle='--')
    plt.plot(epochs_range, valid_losses, label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(filename)
    plt.close() 
    print(f"Loss plot saved to '{filename}'")


if __name__ == '__main__':
    # --- Suppress LPIPS 'pretrained' warnings ---
    warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", "Arguments other than a weight enum or `None` for 'weights' are deprecated")
    # ---

    # Check if data directories exist
    if not os.path.exists(TRAIN_HR_DIR) or not os.path.exists(VALID_HR_DIR):
        print("Error: Training or validation data directory not found.")
        print(f"Please check if '{TRAIN_HR_DIR}' and '{VALID_HR_DIR}' exist.")
    else:
        main()
