import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def differentiable_jpeg(image, quality=75):
    """
    A simplified implementation of differentiable JPEG.
    Note: This is an approximation; true differentiable JPEG is more complex.
    This version omits DCT/IDCT and simulates quantization directly in pixel space.
    
    :param image: Input image tensor, range [-1, 1].
    :param quality: JPEG quality factor (unused, placeholder for interface).
    :return: Simulated JPEG-compressed image.
    """
    # 1. Convert image from [-1, 1] to [0, 255]
    image_uint8 = ((image + 1) / 2) * 255.0

    # 2. Simulate quantization: divide, round, then multiply back.
    # This step is non-differentiable, so we use a straight-through estimator (STE).
    quantization_step = 10 # Simplified quantization step
    quantized = torch.round(image_uint8 / quantization_step) * quantization_step
    
    # STE: In the backward pass, the gradient passes through the round op unchanged.
    quantized = image_uint8 + (quantized - image_uint8).detach()

    # 3. Convert image back to [-1, 1]
    jpeg_image = (quantized / 255.0) * 2.0 - 1.0

    return jpeg_image
