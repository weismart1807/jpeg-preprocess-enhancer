import torch
import torch.nn as nn
import lpips
from torchvision import models
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.vgg_layers = vgg[:36].eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_tensor, target_tensor):
        # 輸入範圍假設為 [-1, 1]，轉換為 [0, 1]
        input_tensor = (input_tensor + 1) / 2.0
        target_tensor = (target_tensor + 1) / 2.0
        
        # 正規化為 ImageNet 統計數據
        input_tensor = (input_tensor - self.mean) / self.std
        target_tensor = (target_tensor - self.mean) / self.std

        if self.resize:
            input_tensor = self.transform(input_tensor, mode='bilinear', size=(224, 224), align_corners=False)
            target_tensor = self.transform(target_tensor, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input_tensor
        y = target_tensor
        for i, block in enumerate(self.vgg_layers):
            x = block(x)
            y = block(y)
            # 選擇特定的層來計算 Loss (例如 ReLU 層)
            if i in [3, 8, 17, 26, 35]: 
                loss += nn.functional.l1_loss(x, y)
        return loss

class TotalLoss(nn.Module):
    def __init__(self, device):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss().to(device)
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

    def forward(self, predicted, target):
        l1 = self.l1_loss(predicted, target)
        vgg = self.vgg_loss(predicted, target)
        lpips_val = self.lpips_loss(predicted, target).mean()

        # 調整權重：讓 L1 佔比較重一點，幫助初期收斂
        total = l1 + 0.1 * vgg + 0.5 * lpips_val
        return total, l1, vgg, lpips_val

def get_color_reduced_image(image_tensor, n_colors):
    """
    使用 GMM 進行顏色量化 (增強穩定版)。
    """
    batch_size, channels, h, w = image_tensor.shape
    reduced_images = []
    
    # 將 tensor 移回 CPU 並轉為 numpy
    images_np = image_tensor.detach().cpu().numpy()
    
    for i in range(batch_size):
        img = images_np[i] # (3, H, W)
        # 反正規化 [-1, 1] -> [0, 1]
        img = (img * 0.5 + 0.5)
        img = np.clip(img, 0, 1)
        
        # (3, H, W) -> (H*W, 3)
        pixels = img.transpose(1, 2, 0).reshape(-1, 3)

        # 加入微量雜訊防止 GMM 在平坦區域崩潰
        noise = np.random.normal(0, 1e-4, pixels.shape) 
        pixels_noisy = pixels + noise

        # 使用 Context Manager 忽略 GMM 的收斂警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            try:
                # 使用較少的迭代次數以加速，並設定隨機種子
                # reg_covar=1e-3: 增加協方差的正則化項，這是解決 "degenerate data" 最有效的方法
                gmm = GaussianMixture(
                    n_components=n_colors, 
                    random_state=42, 
                    max_iter=10, 
                    n_init=1,
                    reg_covar=1e-3
                ).fit(pixels_noisy)
                
                labels = gmm.predict(pixels) # 用原始像素預測
                reduced_pixels = gmm.means_[labels]
            except Exception:
                # 如果 GMM 仍然失敗，退回到簡單的平均顏色 (Fallback)
                reduced_pixels = pixels # 保持原樣或做簡單量化
        
        reduced_img = reduced_pixels.reshape(h, w, 3)
        
        # 轉回 PyTorch Tensor 並正規化至 [-1, 1]
        reduced_img_tensor = torch.from_numpy(reduced_img.transpose(2, 0, 1)).float()
        reduced_img_tensor = (reduced_img_tensor * 2.0) - 1.0
        reduced_images.append(reduced_img_tensor)

    return torch.stack(reduced_images).to(image_tensor.device)