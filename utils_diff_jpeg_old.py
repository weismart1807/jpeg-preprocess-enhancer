import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Constants from Shin & Song paper Appendix A ---
# These are defined on CPU initially

# RGB <-> YCbCr conversion matrix for [0, 255] range
rgb_to_ycrcb_matrix_shin = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
]).float()
ycrcb_offset_shin = torch.tensor([0., 128., 128.]).float()

ycrcb_to_rgb_matrix_shin = torch.tensor([
    [1., 0., 1.402],
    [1., -0.344136, -0.714136], # Paper PDF seems to have an error, using standard value here
    [1., 1.772, 0.]
]).float()

# DCT-related matrices (Appendix A formula)
def create_dct_matrix_shin(N=8):
    """Create NxN DCT-II matrix"""
    matrix = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            if k == 0:
                matrix[k, n] = 1.0 / np.sqrt(N)
            else:
                matrix[k, n] = np.sqrt(2.0 / N) * np.cos((np.pi * k * (2 * n + 1)) / (2.0 * N))
    return matrix

dct_matrix_8x8_shin = create_dct_matrix_shin(8)
idct_matrix_8x8_shin = dct_matrix_8x8_shin.t() # IDCT is the transpose of DCT

# Standard Quantization Tables Qy, Qc (Appendix A)
y_quantization_table_shin = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]).float().view(1, 1, 8, 8) # Add batch and channel dimensions for broadcasting

c_quantization_table_shin = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]).float().view(1, 1, 8, 8)

# Function from quality q to scale factor s (Appendix A formula, note: different from necla-ml)
def quality_to_scale_factor_shin(quality):
    """Calculate quantization table scale factor s from Shin & Song paper"""
    # quality is already a tensor on the correct device here
    quality = torch.clamp(quality, 1, 100).float()
    s = torch.where(quality < 50, 5000 / quality, 200 - 2 * quality)
    return s / 100.0

# --- Differentiable rounding approximation proposed by Shin & Song ---
def diff_round_shin(x):
    """ Differentiable approximation from Shin & Song paper Section 3 """
    floor_x = torch.floor(x)
    return floor_x + torch.pow(x - floor_x, 3)

# --- Helper Functions ---
def image_to_blocks(image_channel):
    """Convert single-channel image (B, 1, H, W) to 8x8 blocks (B, L, 8, 8)"""
    B, C, H, W = image_channel.shape
    unfold = nn.Unfold(kernel_size=8, stride=8)
    blocks_unfolded = unfold(image_channel) # (B, 64, L)
    L = blocks_unfolded.shape[2]
    blocks = blocks_unfolded.permute(0, 2, 1).view(B, L, 8, 8) # (B, L, 8, 8)
    return blocks

def blocks_to_image(blocks, H, W):
    """Combine 8x8 blocks (B, L, 8, 8) back to a single-channel image (B, 1, H, W)"""
    B, L, _, _ = blocks.shape
    blocks_unfolded = blocks.view(B, L, 64).permute(0, 2, 1) # (B, 64, L)
    fold = nn.Fold(output_size=(H, W), kernel_size=8, stride=8)
    image_channel = fold(blocks_unfolded) # (B, 1, H, W)
    return image_channel

# --- Core Function: Shin & Song's Differentiable JPEG Approximation ---
def differentiable_jpeg(image, quality=10):
    """
    Differentiable JPEG approximation implemented based on Shin & Song (2017) paper.
    :param image: Input image tensor (B, C, H, W), range [-1, 1]. H, W must be multiples of 16.
    :param quality: JPEG quality factor (scalar).
    :return: Simulated JPEG compressed image, range [-1, 1].
    """
    device = image.device # Get the device of the input image
    if not isinstance(quality, torch.Tensor):
        # Move quality to the same device as image
        quality = torch.tensor(float(quality), device=device)
    elif quality.device != device:
         # Ensure quality tensor is on the correct device
         quality = quality.to(device)


    B, C, H, W = image.shape
    if H % 16 != 0 or W % 16 != 0:
         raise ValueError("Image dimensions must be divisible by 16 for chroma subsampling")

    # 1. Convert image from [-1, 1] to [0, 255]
    image_0_255 = (image + 1.0) * 127.5

    # 2. RGB -> YCbCr (using paper's matrix and offset)
    # --- (Important) Move constants to the correct device ---
    matrix_rgb_ycbcr = rgb_to_ycrcb_matrix_shin.to(device)
    offset_ycbcr = ycrcb_offset_shin.to(device).view(1, 3, 1, 1)
    # ---
    
    image_permuted = image_0_255.permute(0, 2, 3, 1) # (B, H, W, C)
    ycbcr_permuted = torch.matmul(image_permuted, matrix_rgb_ycbcr.t())
    ycbcr = ycbcr_permuted.permute(0, 3, 1, 2) + offset_ycbcr # (B, C, H, W)
    
    y = ycbcr[:, 0:1, :, :]
    cb = ycbcr[:, 1:2, :, :]
    cr = ycbcr[:, 2:3, :, :]

    # 3. Chroma subsampling
    cb_downsampled = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr_downsampled = F.avg_pool2d(cr, kernel_size=2, stride=2)

    # 4. Split into 8x8 blocks
    y_blocks = image_to_blocks(y)
    cb_blocks = image_to_blocks(cb_downsampled)
    cr_blocks = image_to_blocks(cr_downsampled)

    # 5. DCT
    # --- (Important) Move constants to the correct device ---
    dct_matrix = dct_matrix_8x8_shin.to(device)
    idct_matrix = idct_matrix_8x8_shin.to(device)
    # ---

    y_dct = dct_matrix @ (y_blocks - 128.0) @ idct_matrix # Use idct_matrix (transpose) on the right for DCT
    cb_dct = dct_matrix @ (cb_blocks - 128.0) @ idct_matrix
    cr_dct = dct_matrix @ (cr_blocks - 128.0) @ idct_matrix


    # 6. Quantization
    scale_factor = quality_to_scale_factor_shin(quality) # s is already on the correct device
    # --- (Important) Move constants to the correct device ---
    y_q_table_base = y_quantization_table_shin.to(device)
    c_q_table_base = c_quantization_table_shin.to(device)
    # ---

    # Calculate scaled tables
    y_q_table = y_q_table_base * scale_factor
    c_q_table = c_q_table_base * scale_factor
    
    # Ensure min value is 1
    y_q_table = torch.clamp(y_q_table, min=1.0)
    c_q_table = torch.clamp(c_q_table, min=1.0)

    # Apply differentiable rounding
    y_quantized = diff_round_shin(y_dct / y_q_table)
    cb_quantized = diff_round_shin(cb_dct / c_q_table)
    cr_quantized = diff_round_shin(cr_dct / c_q_table)

    # 7. Dequantization
    y_dequantized = y_quantized * y_q_table
    cb_dequantized = cb_quantized * c_q_table
    cr_dequantized = cr_quantized * c_q_table

    # 8. IDCT
    y_idct = idct_matrix @ y_dequantized @ dct_matrix + 128.0 # Use dct_matrix (transpose) on the right for IDCT
    cb_idct = idct_matrix @ cb_dequantized @ dct_matrix + 128.0
    cr_idct = idct_matrix @ cr_dequantized @ dct_matrix + 128.0

    # 9. Merge blocks
    y_reconstructed = blocks_to_image(y_idct, H, W)
    cb_reconstructed_small = blocks_to_image(cb_idct, H // 2, W // 2)
    cr_reconstructed_small = blocks_to_image(cr_idct, H // 2, W // 2)

    # 10. Chroma upsampling
    cb_upsampled = F.interpolate(cb_reconstructed_small, size=(H, W), mode='bilinear', align_corners=False)
    cr_upsampled = F.interpolate(cr_reconstructed_small, size=(H, W), mode='bilinear', align_corners=False)

    # 11. Merge YCbCr channels
    ycbcr_reconstructed = torch.cat([y_reconstructed, cb_upsampled, cr_upsampled], dim=1)

    # 12. YCbCr -> RGB
    # --- (Important) Move constants to the correct device ---
    matrix_ycbcr_rgb = ycrcb_to_rgb_matrix_shin.to(device)
    # offset_ycbcr_inv = offset_ycbcr.view(1, 3, 1, 1) # Already on device from step 2
    offset_ycbcr_inv = offset_ycbcr # Use the tensor already moved to device
    # ---

    ycbcr_reconstructed_shifted = ycbcr_reconstructed - offset_ycbcr_inv
    ycbcr_rec_permuted = ycbcr_reconstructed_shifted.permute(0, 2, 3, 1) # (B, H, W, C)
    rgb_rec_permuted = torch.matmul(ycbcr_rec_permuted, matrix_ycbcr_rgb.t())
    rgb_reconstructed_0_255 = rgb_rec_permuted.permute(0, 3, 1, 2) # (B, C, H, W)

    # 13. Convert image from [0, 255] back to [-1, 1] and clamp
    image_reconstructed = torch.clamp(rgb_reconstructed_0_255 / 127.5 - 1.0, -1.0, 1.0)

    return image_reconstructed

# --- (Optional) Test code ---
if __name__ == '__main__':
    # Create a dummy input image (Batch=1, RGB, 32x32)
    dummy_image = torch.rand(1, 3, 32, 32) * 2.0 - 1.0 # Range [-1, 1]
    quality = 75

    # Move to CUDA (if available)
    device_to_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device_to_test}")
    dummy_image = dummy_image.to(device_to_test)

    # Run Shin & Song's approximation
    jpeg_approx_image = differentiable_jpeg(dummy_image, quality)

    print("Input image Shape:", dummy_image.shape)
    print("Approx JPEG image Shape:", jpeg_approx_image.shape)
    print("Approx JPEG image range:", torch.min(jpeg_approx_image).item(), "~", torch.max(jpeg_approx_image).item())

    # Check for gradients (if input requires_grad)
    dummy_image.requires_grad = True
    jpeg_approx_image = differentiable_jpeg(dummy_image, quality)
    try:
        jpeg_approx_image.sum().backward()
        print("Gradient calculation successful!")
        # print("Input image gradient sample:", dummy_image.grad[0, 0, 0, :5])
    except Exception as e:
        print("Gradient calculation failed:", e)
