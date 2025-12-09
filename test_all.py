import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
from tqdm import tqdm

# Import your modules
from model import UNet
from utils import get_color_reduced_image

# --- User Configuration ---
MODEL_PATH = 'best_model.pth'   # Path to your trained model weights
TEST_QUALITY = 10               # JPEG quality for testing (e.g., 10)
INPUT_FOLDER = './data/test'    # Folder containing test images
NUM_COLORS = 4                  # Number of representative colors
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PSEUDO_CONTOUR_SUPPRESSION = False # Set to True if needed

# --- Pseudo-Contour Suppression Model (Optional) ---
def apply_pseudo_contour_suppression(image_bgr):
    """
    Applies the pseudo-contour suppression model using OpenCV.
    """
    # 1. Convert to YCrCb, process Y channel only
    try:
        image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    except cv2.error:
        image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        
    y_channel, cr_channel, cb_channel = cv2.split(image_ycbcr)
    
    y_channel_original = y_channel.copy()
    y_channel = y_channel.astype(np.uint8)
    
    h, w = y_channel.shape
    total_image_pixels = h * w

    # --- STEP 1: Find low-frequency regions ---
    y_median = cv2.medianBlur(y_channel, 5) 
    edges_step1 = cv2.Canny(y_median, 20, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges_step1, cv2.MORPH_CLOSE, kernel)
    low_freq_mask = cv2.bitwise_not(edges_closed)

    # --- STEP 2: Labeling ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(low_freq_mask, 8, cv2.CV_32S)
    y_homogenized = y_channel.copy()

    # --- STEP 3: Homogenize gradient regions ---
    for i in range(1, num_labels):
        region_mask = (labels == i).astype(np.uint8)
        region_pixels = stats[i, cv2.CC_STAT_AREA]

        if region_pixels > (total_image_pixels * 0.10):
            region_y = cv2.bitwise_and(y_channel, y_channel, mask=region_mask)
            region_y_median = cv2.medianBlur(region_y, 5)
            internal_edges = cv2.Canny(region_y_median, 0, 10)
            internal_edges = cv2.bitwise_and(internal_edges, internal_edges, mask=region_mask)
            num_internal_edge_pixels = np.sum(internal_edges > 0)

            if num_internal_edge_pixels < (region_pixels * 0.03):
                avg_val = cv2.mean(y_channel, mask=region_mask)[0]
                y_homogenized[region_mask == 1] = int(avg_val)

    # --- STEP 3 (cont.): Guided Filter ---
    try:
        radius = 10
        eps = (0.1 * 255)**2 
        y_final = cv2.ximgproc.guidedFilter(guide=y_channel_original, src=y_homogenized, radius=radius, eps=eps, dDepth=-1)
    except Exception as e:
        y_final = y_homogenized # Fallback

    final_ycbcr = cv2.merge([y_final.astype(np.uint8), cr_channel, cb_channel])
    final_bgr = cv2.cvtColor(final_ycbcr, cv2.COLOR_YCrCb2BGR)
    
    return final_bgr

# --- Helper Functions ---
def numpy_to_lpips_tensor(image_rgb, device):
    """Converts [0, 255] HWC numpy image to [-1, 1] BCHW tensor for LPIPS"""
    image_float = image_rgb.astype(np.float32) / 255.0
    image_chw = torch.from_numpy(image_float).permute(2, 0, 1)
    image_tensor = (image_chw * 2.0) - 1.0
    return image_tensor.unsqueeze(0).to(device)

def eval_single_image(model, lpips_model, image_path, quality, device):
    """
    Evaluates a single image and returns metrics.
    """
    # 1. Load and Preprocess
    image_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    original_image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # 2. Model Prediction
    with torch.no_grad():
        color_reduced_image = get_color_reduced_image(original_image_tensor, NUM_COLORS)
        color_residuals = model(color_reduced_image, original_image_tensor)
        predicted_image = color_reduced_image + color_residuals
        predicted_image = torch.clamp(predicted_image, -1.0, 1.0)

    # Convert to OpenCV BGR [0, 255]
    output_image_np = (predicted_image.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255
    output_image_np = output_image_np.transpose(1, 2, 0).astype(np.uint8)
    output_image_bgr_final = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

    # Optional: Pseudo-Contour Suppression
    if PSEUDO_CONTOUR_SUPPRESSION:
        output_image_bgr_final = apply_pseudo_contour_suppression(output_image_bgr_final)

    # 3. JPEG Compression Simulation
    original_image_bgr = cv2.imread(image_path)
    # Resize original if dimensions mismatch (e.g. padding in model)
    if original_image_bgr.shape[:2] != output_image_bgr_final.shape[:2]:
         original_image_bgr = cv2.resize(original_image_bgr, (output_image_bgr_final.shape[1], output_image_bgr_final.shape[0]))

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    # Standard JPEG
    _, buf_std = cv2.imencode('.jpg', original_image_bgr, jpeg_params)
    std_jpeg_bgr = cv2.imdecode(buf_std, cv2.IMREAD_COLOR)

    # Proposed JPEG
    _, buf_prop = cv2.imencode('.jpg', output_image_bgr_final, jpeg_params)
    prop_jpeg_bgr = cv2.imdecode(buf_prop, cv2.IMREAD_COLOR)

    # 4. Calculate Metrics (Convert to RGB)
    original_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    std_rgb = cv2.cvtColor(std_jpeg_bgr, cv2.COLOR_BGR2RGB)
    prop_rgb = cv2.cvtColor(prop_jpeg_bgr, cv2.COLOR_BGR2RGB)

    # PSNR & SSIM
    psnr_std = peak_signal_noise_ratio(original_rgb, std_rgb, data_range=255)
    ssim_std = structural_similarity(original_rgb, std_rgb, channel_axis=-1, data_range=255)
    
    psnr_prop = peak_signal_noise_ratio(original_rgb, prop_rgb, data_range=255)
    ssim_prop = structural_similarity(original_rgb, prop_rgb, channel_axis=-1, data_range=255)

    # LPIPS
    std_tensor = numpy_to_lpips_tensor(std_rgb, device)
    prop_tensor = numpy_to_lpips_tensor(prop_rgb, device)
    
    with torch.no_grad():
        lpips_std = lpips_model(original_image_tensor, std_tensor).item()
        lpips_prop = lpips_model(original_image_tensor, prop_tensor).item()

    return (psnr_std, ssim_std, lpips_std, psnr_prop, ssim_prop, lpips_prop)

def main():
    warnings.filterwarnings("ignore")
    print(f"Using Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Quality: {TEST_QUALITY}")

    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found.")
        return

    print("Loading UNet and LPIPS...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder {INPUT_FOLDER} not found.")
        return
        
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    if len(image_files) == 0:
        print("No images found in folder.")
        return

    print(f"Found {len(image_files)} images. Starting evaluation...\n")

    total_metrics = np.zeros(6) 
    results_log = []
    
    # Progress bar loop
    pbar = tqdm(image_files, desc="Evaluating")
    success_count = 0 

    for i, img_name in enumerate(pbar):
        img_path = os.path.join(INPUT_FOLDER, img_name)
        
        try:
            # --- Core Evaluation Logic ---
            metrics = eval_single_image(model, lpips_model, img_path, TEST_QUALITY, DEVICE)
            metrics_np = np.array(metrics)
            
            # Accumulate results
            total_metrics += metrics_np
            success_count += 1
            
            # Calculate current average
            current_avg = total_metrics / success_count
            
            # Log details
            log_str = (f"{img_name}: \n"
                       f"  STD  -> PSNR: {metrics[0]:.2f}, SSIM: {metrics[1]:.4f}, LPIPS: {metrics[2]:.4f}\n"
                       f"  PROP -> PSNR: {metrics[3]:.2f}, SSIM: {metrics[4]:.4f}, LPIPS: {metrics[5]:.4f}")
            results_log.append(log_str)
            
            # Update progress bar
            pbar.set_postfix({
                'Avg_PSNR': f"{current_avg[3]:.2f}", 
                'Avg_LPIPS': f"{current_avg[5]:.4f}"
            })

            # --- Explicit Memory Cleanup ---
            del metrics
            del metrics_np
            torch.cuda.empty_cache()

        except RuntimeError as e:
            # Error handling for OOM
            if "out of memory" in str(e):
                print(f"\n[Warning] OOM on {img_name}. Skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    # --- Final Report ---
    if success_count > 0:
        final_avg = total_metrics / success_count
        
        summary = (
            "\n" + "="*40 + "\n"
            f" FINAL RESULTS (Avg of {success_count} images)\n"
            f" Model: {MODEL_PATH} | Quality: {TEST_QUALITY}\n"
            + "-"*40 + "\n"
            f" Standard JPEG:\n"
            f"   PSNR : {final_avg[0]:.4f}\n"
            f"   SSIM : {final_avg[1]:.4f}\n"
            f"   LPIPS: {final_avg[2]:.4f}\n"
            + "-"*20 + "\n"
            f" Proposed Method:\n"
            f"   PSNR : {final_avg[3]:.4f}\n"
            f"   SSIM : {final_avg[4]:.4f}\n"
            f"   LPIPS: {final_avg[5]:.4f}\n"
            + "="*40 + "\n"
        )

        print(summary)
        
        with open("results.txt", "w") as f:
            f.write(summary)
            f.write("\nDetailed Results:\n")
            for log in results_log:
                f.write(log + "\n")
        print("Results saved to 'results.txt'.")
    else:
        print("No images were successfully evaluated.")

if __name__ == '__main__':
    main()