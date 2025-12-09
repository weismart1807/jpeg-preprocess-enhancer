import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import os
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings # For suppressing warnings

# From other files
from model import UNet
from utils import get_color_reduced_image

# --- Pseudo-Contour Suppression Model (from Paper Fig. 9) ---
def apply_pseudo_contour_suppression(image_bgr):
    """
    Applies the pseudo-contour suppression model from Shoda et al.
    This is an OpenCV-based signal processing pipeline.
    """
    print("    (Step 2a: Converting to YCrCb space...)")
    # 1. Convert to YCrCb, we only process the Y (luma) channel
    try:
        image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    except cv2.error:
        image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        
    y_channel, cr_channel, cb_channel = cv2.split(image_ycbcr)
    
    # Keep original Y channel as guide for Guided Filter
    y_channel_original = y_channel.copy()
    y_channel = y_channel.astype(np.uint8)
    
    h, w = y_channel.shape
    total_image_pixels = h * w

    # --- STEP 1: Find low-frequency regions ---
    print("    (Step 2b: Finding low-frequency regions...)")
    # Denoise (Paper 4.2)
    y_median = cv2.medianBlur(y_channel, 5) 
    # Edge detection (Th_min=20, Th_max=30)
    edges_step1 = cv2.Canny(y_median, 20, 30)
    # Closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges_step1, cv2.MORPH_CLOSE, kernel)
    # Invert to get low-frequency mask
    low_freq_mask = cv2.bitwise_not(edges_closed)

    # --- STEP 2: Labeling ---
    print("    (Step 2c: Labeling regions...)")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(low_freq_mask, 8, cv2.CV_32S)

    # Prepare a Y channel for modification
    y_homogenized = y_channel.copy()

    # --- STEP 3: Homogenize gradient regions ---
    print("    (Step 2d: Homogenizing gradient regions...)")
    # Skip label 0 (background, i.e., high-frequency regions)
    for i in range(1, num_labels):
        region_mask = (labels == i).astype(np.uint8)
        region_pixels = stats[i, cv2.CC_STAT_AREA]

        # Condition 2: Region must be > 10% of total image pixels (Paper 4.2)
        if region_pixels > (total_image_pixels * 0.10):
            # Extract only the Y channel for this region
            region_y = cv2.bitwise_and(y_channel, y_channel, mask=region_mask)
            
            # Use a more sensitive Canny to detect internal edges (Th_min=0, Th_max=10)
            region_y_median = cv2.medianBlur(region_y, 5)
            internal_edges = cv2.Canny(region_y_median, 0, 10)
            # Ensure internal edges are only within the mask
            internal_edges = cv2.bitwise_and(internal_edges, internal_edges, mask=region_mask)
            num_internal_edge_pixels = np.sum(internal_edges > 0)

            # Condition 1: Internal edges < 3% of region pixels (Paper 4.2)
            if num_internal_edge_pixels < (region_pixels * 0.03):
                # Determined as "gradient region", apply homogenization
                avg_val = cv2.mean(y_channel, mask=region_mask)[0]
                y_homogenized[region_mask == 1] = int(avg_val)

    # --- STEP 3 (cont.): Guided Filter ---
    print("    (Step 2e: Applying Guided Filter...)")
    try:
        # Use original Y as guide, smooth the homogenized Y
        # This smooths the edges of the replaced regions, making them more natural
        radius = 10
        eps = (0.1 * 255)**2 # (0.1*L)^2, L=dynamic range
        
        # Must use opencv-python-contrib
        y_final = cv2.ximgproc.guidedFilter(guide=y_channel_original, src=y_homogenized, radius=radius, eps=eps, dDepth=-1)
    
    except Exception as e:
        print("--- WARNING ---")
        print(f"Guided Filter failed: {e}")
        print("Guided Filter step was skipped. Edges might look unnatural.")
        print("-------------------------")
        y_final = y_homogenized # Fallback: use the version without smoothed edges

    # 4. Merge channels and convert back to BGR
    print("    (Step 2f: Converting back to BGR...)")
    final_ycbcr = cv2.merge([y_final.astype(np.uint8), cr_channel, cb_channel])
    final_bgr = cv2.cvtColor(final_ycbcr, cv2.COLOR_YCrCb2BGR)
    
    return final_bgr
# --- End of function ---

# --- LPIPS evaluation helper ---
def numpy_to_lpips_tensor(image_rgb, device):
    """Converts [0, 255] HWC_uint8 numpy image to [-1, 1] BCHW_float tensor for LPIPS"""
    image_float = image_rgb.astype(np.float32) / 255.0
    image_chw = torch.from_numpy(image_float).permute(2, 0, 1)
    image_tensor = (image_chw * 2.0) - 1.0
    return image_tensor.unsqueeze(0).to(device)
# --- End of function ---


def main(args, NUM_REPRESENTATIVE_COLORS):
    """
    Main execution function to load the model and test a single image.
    """
    # --- Suppress LPIPS 'pretrained' warnings ---
    warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", "Arguments other than a weight enum or `None` for 'weights' are deprecated")
    # ---

    # --- 1. Setup and Model Loading ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Load LPIPS model for evaluation ---
    print("Loading LPIPS VGG model (for evaluation)...")
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'. Please train first.")
        return

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Image Loading and Preprocessing ---
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at '{args.input}'.")
        return

    image_pil = Image.open(args.input).convert('RGB')
    
    # Convert to Tensor and normalize to [-1, 1] (for LPIPS reference)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    original_image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)

    # --- 3. Pre-processing Step 1: Color Transform (U-Net) ---
    print("Running pre-processing (Step 1: Color Transform)...")
    with torch.no_grad():
        color_reduced_image = get_color_reduced_image(original_image_tensor, NUM_REPRESENTATIVE_COLORS)
        color_residuals = model(color_reduced_image, original_image_tensor)
        predicted_image = color_reduced_image + color_residuals
        predicted_image = torch.clamp(predicted_image, -1.0, 1.0)
    
    # Convert prediction to OpenCV BGR format [0, 255]
    output_image_np = (predicted_image.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255
    output_image_np = output_image_np.transpose(1, 2, 0).astype(np.uint8)
    output_image_bgr_final = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)
    
    ''' if you want to add Pseudo-Contour Suppression 
    # --- 3.5. Pre-processing Step 2: Pseudo-Contour Suppression ---
    print("Running pre-processing (Step 2: Pseudo-Contour Suppression)...")
    output_image_bgr_final = apply_pseudo_contour_suppression(output_image_bgr_final)
    print("Pre-processing finished.")
    '''
    
    # --- 4. Apply JPEG Compression ---
    original_image_bgr = cv2.imread(args.input)
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    
    std_jpeg_path = 'temp_standard_jpeg.jpg'
    proposed_jpeg_path = 'temp_proposed_jpeg.jpg'
    
    cv2.imwrite(std_jpeg_path, original_image_bgr, jpeg_params)
    # Save the image processed by the full two-step pipeline
    cv2.imwrite(proposed_jpeg_path, output_image_bgr_final, jpeg_params) 
    print(f"Generated JPEG compressed images with quality={args.quality}.")

    # --- 5. Display Results & Calculate Metrics ---
    # Read the just-saved images for display
    std_jpeg_img_bgr = cv2.imread(std_jpeg_path)
    proposed_jpeg_img_bgr = cv2.imread(proposed_jpeg_path)

    # Convert BGR to RGB (Numpy, 0-255) for matplotlib and PSNR/SSIM
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    std_jpeg_rgb = cv2.cvtColor(std_jpeg_img_bgr, cv2.COLOR_BGR2RGB)
    proposed_jpeg_rgb = cv2.cvtColor(proposed_jpeg_img_bgr, cv2.COLOR_BGR2RGB)
    
    # --- Calculate PSNR and SSIM (using Numpy 0-255) ---
    print("Calculating PSNR and SSIM...")
    # data_range=255 informs SSIM our data is 0-255
    psnr_std = peak_signal_noise_ratio(original_image_rgb, std_jpeg_rgb, data_range=255)
    ssim_std = structural_similarity(original_image_rgb, std_jpeg_rgb, channel_axis=-1, data_range=255)
    
    psnr_prop = peak_signal_noise_ratio(original_image_rgb, proposed_jpeg_rgb, data_range=255)
    ssim_prop = structural_similarity(original_image_rgb, proposed_jpeg_rgb, channel_axis=-1, data_range=255)

    # --- Calculate LPIPS (using Tensor -1 to 1) ---
    print("Calculating LPIPS...")
    # Convert Numpy images to Tensors for LPIPS
    std_jpeg_tensor = numpy_to_lpips_tensor(std_jpeg_rgb, DEVICE)
    proposed_jpeg_tensor = numpy_to_lpips_tensor(proposed_jpeg_rgb, DEVICE)
    
    with torch.no_grad():
        lpips_std = lpips_model(original_image_tensor, std_jpeg_tensor).item()
        lpips_prop = lpips_model(original_image_tensor, proposed_jpeg_tensor).item()

    print("Metric calculation complete.")

    # --- Create titles with metrics ---
    std_title = f'Standard JPEG (Q: {args.quality})\n' \
                f'PSNR: {psnr_std:.2f} dB | SSIM: {ssim_std:.4f} | LPIPS: {lpips_std:.4f} ↓'
    
    prop_title = f'Enhanced (All Steps) + JPEG (Q: {args.quality})\n' \
                 f'PSNR: {psnr_prop:.2f} dB | SSIM: {ssim_prop:.4f} | LPIPS: {lpips_prop:.4f} ↓'

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=600)
    axes[0].imshow(original_image_rgb)
    axes[0].set_title('Original', fontsize=2)
    axes[0].axis('off')

    axes[1].imshow(std_jpeg_rgb)
    axes[1].set_title(std_title, fontsize=2)
    axes[1].axis('off')

    axes[2].imshow(proposed_jpeg_rgb)
    axes[2].set_title(prop_title, fontsize=2)
    axes[2].axis('off')

    plt.tight_layout(pad=2.0)
    plt.show()

    # --- 6. Cleanup temp files ---
    os.remove(std_jpeg_path)
    os.remove(proposed_jpeg_path)


if __name__ == '__main__':
    # --- Suppress LPIPS 'pretrained' warnings ---
    warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", "Arguments other than a weight enum or `None` for 'weights' are deprecated")
    # ---

    parser = argparse.ArgumentParser(description='Test script for JPEG enhancement using a trained model.')
    parser.add_argument('--input', type=str, default="./data/test/00015_TE_3680x2456.png", help='Path to the input image.')
    # parser.add_argument('--model', type=str, default='best_model_new-diff_guss01_color04.pth', help='Path to the trained model weights.')
    parser.add_argument('--model', type=str, default='best_model_new-diff_guss03_color04.pth', help='Path to the trained model weights.')
    # parser.add_argument('--model', type=str, default='best_model_new-diff.pth', help='Path to the trained model weights.')
    parser.add_argument('--quality', type=int, default=1, choices=range(0, 101), metavar="[0-100]", help='JPEG compression quality (0-100).')
    
    args = parser.parse_args()
    NUM_REPRESENTATIVE_COLORS = 4

    main(args, NUM_REPRESENTATIVE_COLORS)

