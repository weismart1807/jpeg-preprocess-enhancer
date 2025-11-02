import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings

# --- (Important) Import the necla-ml library ---
try:
    from diff_jpeg import diff_jpeg_coding
except ImportError:
    print("Error: 'diff-jpeg' library not found.")
    print("Please install it in your (DIP_env) environment: pip install diff-jpeg")
    exit()

# --- Configure Matplotlib Font (for '↓' symbol) ---
try:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    try:
        matplotlib.rcParams['font.sans-serif'] = ['DFKai-SB']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("Warning: Could not find 'Microsoft JhengHei' or 'DFKai-SB' font.")
        print("Non-ASCII characters in the plot title might not display correctly.")

# --- Helper Functions for DiffJPEG ---

def ensure_divisible_by_16(image_tensor):
    """
    Pads the image tensor so its H and W dimensions are multiples of 16.
    (Required by necla-ml/Diff-JPEG for chroma subsampling)
    Returns (padded_image, original_H, original_W)
    """
    B, C, H, W = image_tensor.shape
    pad_h = (16 - (H % 16)) % 16
    pad_w = (16 - (W % 16)) % 16
    
    if pad_h > 0 or pad_w > 0:
        # (pad_left, pad_right, pad_top, pad_bottom)
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), 'replicate')
        
    return image_tensor, H, W

def differentiable_jpeg_wrapper(image_tensor_neg1_1, quality):
    """
    A safe wrapper for necla-ml/Diff-JPEG.
    Handles: [-1, 1] range, 16x padding, and Quality tensor.
    """
    device = image_tensor_neg1_1.device
    B, C, H_orig, W_orig = image_tensor_neg1_1.shape

    # 1. Pad to multiple of 16
    image_padded, H_orig, W_orig = ensure_divisible_by_16(image_tensor_neg1_1)

    # 2. Convert range from [-1, 1] to [0, 1] (required by necla-ml)
    image_0_1 = (image_padded + 1.0) / 2.0

    # 3. Prepare Quality tensor
    quality_tensor = torch.full((B,), float(quality), device=device, dtype=torch.float)

    # 4. Call Diff-JPEG (ste=False uses the "soft simulation")
    jpeg_image_0_1 = diff_jpeg_coding(image_0_1, quality_tensor, ste=False)

    # 5. Crop back to original size
    jpeg_image_0_1_cropped = jpeg_image_0_1[:, :, :H_orig, :W_orig]
    
    # 6. Convert back to [-1, 1]
    jpeg_image_neg1_1 = (jpeg_image_0_1_cropped * 2.0) - 1.0
    
    return jpeg_image_neg1_1

# --- LPIPS Helper Function ---
def numpy_to_lpips_tensor(image_rgb, device):
    """Converts [0, 255] HWC_uint8 numpy image to [-1, 1] BCHW_float tensor needed by LPIPS"""
    image_float = image_rgb.astype(np.float32) / 255.0
    image_chw = torch.from_numpy(image_float).permute(2, 0, 1)
    image_tensor = (image_chw * 2.0) - 1.0
    return image_tensor.unsqueeze(0).to(device)
# ---

def main(args):
    """
    Main execution function:
    1. Load image
    2. Calculate distortion from DiffJPEG
    3. Create "pre-compensated" image
    4. Compress both "pre-compensated" and "original" images
    5. Compare the results
    """
    # --- 1. Setup ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load LPIPS evaluation model
    print("Loading LPIPS VGG model (for evaluation)...")
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)
    lpips_model.eval()

    # --- 2. Image Loading ---
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at '{args.input}'.")
        return
    image_pil = Image.open(args.input).convert('RGB')

    # Transform to Tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    original_image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    print("Original image loaded.")

    # --- 3. (ERROR) Pre-Compensation (Simple Addition) ---
    print(f"Calculating (flawed) pre-compensation using DiffJPEG (Quality={args.quality})...")
    with torch.no_grad():
        # Step 1: Get the simulated JPEG result
        jpeg_approx_image = differentiable_jpeg_wrapper(original_image_tensor, args.quality)
        
        # Step 2: Calculate distortion (residual)
        distortion = original_image_tensor - jpeg_approx_image
        
        # Step 3: Create the pre-compensated image (I_pre = I + Distortion)
        pre_compensated_tensor = original_image_tensor + distortion
        
        # --- This clamp is the source of the error ---
        pre_compensated_tensor = torch.clamp(pre_compensated_tensor, -1.0, 1.0)
        
    print("Pre-compensation (with clamp) calculated.")

    # --- 4. Prepare for Real Compression ---
    
    # Read original image (OpenCV BGR, 0-255)
    original_image_bgr = cv2.imread(args.input)
    
    # Convert "pre-compensated" tensor (PyTorch RGB, -1..1) to (OpenCV BGR, 0..255)
    pre_comp_rgb_np = (pre_compensated_tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255
    # Clip to 0-255 range AFTER conversion
    pre_comp_rgb_np = np.clip(pre_comp_rgb_np, 0, 255).astype(np.uint8) 
    pre_compensated_bgr = cv2.cvtColor(pre_comp_rgb_np, cv2.COLOR_RGB2BGR)

    
    # --- 5. Perform Real JPEG Compression ---
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    
    std_jpeg_path = 'temp_standard_jpeg.jpg'
    proposed_jpeg_path = 'temp_proposed_jpeg.jpg'
    
    cv2.imwrite(std_jpeg_path, original_image_bgr, jpeg_params)
    cv2.imwrite(proposed_jpeg_path, pre_compensated_bgr, jpeg_params)
    print(f"Generated REAL JPEG compressed images with quality={args.quality}.")
    
    # --- 6. Display Comparison & Calculate Metrics ---
    std_jpeg_img_bgr = cv2.imread(std_jpeg_path)
    proposed_jpeg_img_bgr = cv2.imread(proposed_jpeg_path)

    # Convert to RGB (Numpy, 0-255)
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    std_jpeg_rgb = cv2.cvtColor(std_jpeg_img_bgr, cv2.COLOR_BGR2RGB)
    proposed_jpeg_rgb = cv2.cvtColor(proposed_jpeg_img_bgr, cv2.COLOR_BGR2RGB)

    # --- Calculate PSNR and SSIM ---
    print("Calculating PSNR and SSIM...")
    psnr_std = peak_signal_noise_ratio(original_image_rgb, std_jpeg_rgb, data_range=255)
    ssim_std = structural_similarity(original_image_rgb, std_jpeg_rgb, channel_axis=-1, data_range=255, win_size=7)
    
    psnr_prop = peak_signal_noise_ratio(original_image_rgb, proposed_jpeg_rgb, data_range=255)
    ssim_prop = structural_similarity(original_image_rgb, proposed_jpeg_rgb, channel_axis=-1, data_range=255, win_size=7)

    # --- Calculate LPIPS ---
    print("Calculating LPIPS...")
    std_jpeg_tensor = numpy_to_lpips_tensor(std_jpeg_rgb, DEVICE)
    proposed_jpeg_tensor = numpy_to_lpips_tensor(proposed_jpeg_rgb, DEVICE)
    
    with torch.no_grad():
        lpips_std = lpips_model(original_image_tensor, std_jpeg_tensor).item()
        lpips_prop = lpips_model(original_image_tensor, proposed_jpeg_tensor).item()

    print("Metric calculation finished.")

    # --- Create Titles ---
    std_title = f'Standard JPEG (Q: {args.quality})\n' \
                f'PSNR: {psnr_std:.2f} dB | SSIM: {ssim_std:.4f} | LPIPS: {lpips_std:.4f} ↓'
    
    prop_title = f'Pre-Compensated (Non-Learning) + JPEG (Q: {args.quality})\n' \
                 f'PSNR: {psnr_prop:.2f} dB | SSIM: {ssim_prop:.4f} | LPIPS: {lpips_prop:.4f} ↓'
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=600)
    title_fontsize = 2 # As requested

    axes[0].imshow(original_image_rgb)
    axes[0].set_title('Original', fontsize=title_fontsize)
    axes[0].axis('off')

    axes[1].imshow(std_jpeg_rgb)
    axes[1].set_title(std_title, fontsize=title_fontsize)
    axes[1].axis('off')

    axes[2].imshow(proposed_jpeg_rgb)
    axes[2].set_title(prop_title, fontsize=title_fontsize)
    axes[2].axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()

    # --- 7. Cleanup ---
    os.remove(std_jpeg_path)
    os.remove(proposed_jpeg_path)


if __name__ == '__main__':
    # --- Suppress LPIPS warnings ---
    warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", "Arguments other than a weight enum or `None` for 'weights' are deprecated")

    parser = argparse.ArgumentParser(description='Test Non-Learning Pre-Compensation using Diff-JPEG.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--quality', type=int, default=10, choices=range(0, 101), help='JPEG compression quality (0-100).')
    
    args = parser.parse_args()
    main(args)

