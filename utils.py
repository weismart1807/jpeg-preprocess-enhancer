import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models
from sklearn.mixture import GaussianMixture
import numpy as np # Added for get_color_reduced_image

class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss.
    Uses a pre-trained VGG19 network to extract features and compute L1 loss
    between feature maps.
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Use the first few convolutional layers of VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg_layers = vgg[:35].eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Convert input images from [-1, 1] to [0, 1]
        input = (input + 1) / 2
        target = (target + 1) / 2

        # Normalize using VGG's mean and std
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        features_input = self.vgg_layers(input)
        features_target = self.vgg_layers(target)

        return F.l1_loss(features_input, features_target)

def get_color_reduced_image(image_tensor, n_colors):
    """
    Generates a color-reduced image using Gaussian Mixture Model (GMM).
    Note: This operation runs on the CPU sample-by-sample and can be a 
    performance bottleneck during training.
    """
    batch_size = image_tensor.shape[0]
    reduced_images = []
    
    for i in range(batch_size):
        img = image_tensor[i].detach().cpu().numpy()
        # Reshape from (C, H, W) to (H*W, C) for GMM processing
        img = img.transpose(1, 2, 0)
        h, w, c = img.shape
        pixels = img.reshape(-1, c)

        # Fit GMM model
        gmm = GaussianMixture(n_components=n_colors, random_state=0).fit(pixels)
        
        # Predict labels and replace with cluster centers (means)
        labels = gmm.predict(pixels)
        reduced_pixels = gmm.means_[labels]
        
        # Reshape back to original image shape (H, W, C)
        reduced_img = reduced_pixels.reshape(h, w, c)
        
        # Convert back to (C, H, W) PyTorch tensor
        reduced_img_tensor = torch.from_numpy(reduced_img.transpose(2, 0, 1)).float()
        reduced_images.append(reduced_img_tensor)

    return torch.stack(reduced_images).to(image_tensor.device)


class TotalLoss(nn.Module):
    """
    Calculates the total loss as per the paper: L_total = L1 + L_SSIM + L_VGG.
    Replaces MS-SSIM with LPIPS, as it correlates better with human perception.
    """
    def __init__(self, device):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss().to(device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device) # LPIPS is similar to MS-SSIM

    def forward(self, predicted, target):
        l1 = self.l1_loss(predicted, target)
        vgg = self.vgg_loss(predicted, target)
        lpips_val = self.lpips_loss(predicted, target).mean() # lpips_loss returns a tensor, take the mean

        total = l1 + vgg + lpips_val
        return total, l1, vgg, lpips_val

