import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net architecture based on Table 1 from the Shoda et al. paper.
    """
    def __init__(self):
        super(UNet, self).__init__()

        # --- Encoder ---
        # Conv-1: (H, W, 6) -> (H/2, W/2, 32)
        self.encoder1 = self._conv_block(6, 32, 'encoder1')
        # Conv-2: (H/2, W/2, 32) -> (H/4, W/4, 64)
        self.encoder2 = self._conv_block(32, 64, 'encoder2')
        # Bottleneck: (H/4, W/4, 64) -> (H/8, W/8, 128)
        self.bottleneck = self._conv_block(64, 128, 'bottleneck')

        # --- Decoder ---
        # Deconv-1: (H/8, W/8, 128) -> (H/4, W/4, 64)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Concat(Deconv-1, Conv-2) -> 64+64=128
        self.decoder1 = self._conv_block(128, 32, 'decoder1') 

        # Deconv-2: (H/4, W/4, 32) -> (H/2, W/2, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Concat(Deconv-2, Conv-1) -> 32+32=64
        self.decoder2 = self._conv_block(64, 32, 'decoder2') 

        # Deconv-3: (H/2, W/2, 32) -> (H, W, 32)
        self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # Per Table 1: Concat(Deconv-3, Input image)
        # Input image is 3 channels.
        # 32 (from deconv3) + 3 (original image) = 35
        self.final_conv1 = self._conv_block(35, 6, 'final_conv1') 
        self.final_conv2 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()


    def _conv_block(self, in_channels, out_channels, name):
        """
        Defines a convolutional block with Conv2d, ReLU, BatchNorm.
        Downsampling is achieved with stride=2 in encoder blocks.
        """
        if 'encoder' in name or 'bottleneck' in name:
            # Downsampling block
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Regular block (stride=1)
             return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, color_reduced_image, original_image):
        """
        Forward pass.
        Input is HxWx6 per Table 1 (concatenation of two images).
        """
        x = torch.cat([color_reduced_image, original_image], dim=1) # (B, 6, H, W)

        # Encoder
        enc1 = self.encoder1(x) # -> (B, 32, H/2, W/2)
        enc2 = self.encoder2(enc1) # -> (B, 64, H/4, W/4)
        bottleneck = self.bottleneck(enc2) # -> (B, 128, H/8, W/8)

        # Decoder
        dec1 = self.upconv1(bottleneck) # -> (B, 64, H/4, W/4)
        dec1 = torch.cat([dec1, enc2], dim=1) # Skip connection
        dec1 = self.decoder1(dec1) # -> (B, 32, H/4, W/4)

        dec2 = self.upconv2(dec1) # -> (B, 32, H/2, W/2)
        dec2 = torch.cat([dec2, enc1], dim=1) # Skip connection
        dec2 = self.decoder2(dec2) # -> (B, 32, H/2, W/2)

        dec3 = self.upconv3(dec2) # -> (B, 32, H, W)
        
        # Final concatenation with original 3-channel image per Table 1
        dec3 = torch.cat([dec3, original_image], dim=1) # -> (B, 35, H, W)

        # Final convolution layers
        out = self.final_conv1(dec3)
        out = self.final_conv2(out)
        
        # Tanh activation to constrain output to [-1, 1]
        color_residual = self.tanh(out)

        return color_residual

if __name__ == '__main__':
    # Test model architecture
    model = UNet()
    # Test with B=2, C=3, H=128, W=128
    test_reduced_img = torch.randn(2, 3, 128, 128)
    test_original_img = torch.randn(2, 3, 128, 128)
    output = model(test_reduced_img, test_original_img)
    print(f"Model output shape: {output.shape}") # Should be (2, 3, 128, 128)
