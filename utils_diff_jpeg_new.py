import torch
import torch.nn as nn

# 嘗試導入 diff_jpeg 函式庫
try:
    from diff_jpeg import DiffJPEGCoding
    LIBRARY_AVAILABLE = True
except ImportError:
    DiffJPEGCoding = None
    LIBRARY_AVAILABLE = False
    print("警告: 找不到 'diff_jpeg' 模組。請確保已安裝該函式庫。")

# 建立一個全域的模組實例
_diff_jpeg_module_instance = None

def get_jpeg_module(device):
    """
    Lazy initialization of the DiffJPEGCoding module.
    """
    global _diff_jpeg_module_instance
    if _diff_jpeg_module_instance is None and LIBRARY_AVAILABLE:
        _diff_jpeg_module_instance = DiffJPEGCoding()
    
    if _diff_jpeg_module_instance is not None:
        _diff_jpeg_module_instance.to(device)
        
    return _diff_jpeg_module_instance

def differentiable_jpeg(image_tensor, quality=10.0):
    """
    Differentiable JPEG 的包裝函式。
    
    修正重點：
    將輸入範圍從 [-1, 1] 轉換為 [0, 255]，以符合 Reich et al. 論文與標準 JPEG 量化表的預期。
    """
    if not LIBRARY_AVAILABLE:
        raise ImportError("無法執行 Differentiable JPEG，因為缺少 'diff_jpeg' 模組。")

    # 取得模組實例
    jpeg_module = get_jpeg_module(image_tensor.device)

    # 1. 數值範圍轉換 [重要修改]
    # 原本是轉為 [0, 1]，這會導致量化後訊號消失。
    # 根據論文，Diff JPEG 運作在 [0, 255] 空間 。
    # 先做 clamp 確保數值穩定，再縮放。
    image_clamped = torch.clamp(image_tensor, -1.0, 1.0)
    image_255 = (image_clamped + 1.0) * 127.5
    
    # 2. 準備 Quality 參數
    if isinstance(quality, torch.Tensor):
        jpeg_quality = quality.to(image_tensor.device)
    else:
        jpeg_quality = torch.tensor([float(quality)], device=image_tensor.device)
    
    if jpeg_quality.numel() == 1:
        batch_size = image_tensor.shape[0]
        jpeg_quality = jpeg_quality.repeat(batch_size)
        
    jpeg_quality = jpeg_quality.view(-1)

    # 3. 執行 Differentiable JPEG 編碼
    try:
        # 傳入 [0, 255] 範圍的影像
        image_coded_255 = jpeg_module(image_rgb=image_255, jpeg_quality=jpeg_quality)
    except Exception as e:
        print(f"DiffJPEG 執行錯誤: {e}")
        return image_tensor

    # 4. 數值範圍還原 [重要修改]
    # 將輸出從 [0, 255] 轉回 [-1, 1] 以計算 Loss (Tanh domain)
    image_coded = (image_coded_255 / 127.5) - 1.0
    
    # 額外保險：確保輸出不超過 [-1, 1] (Reich 的方法有時會有些許溢出)
    image_coded = torch.clamp(image_coded, -1.0, 1.0)
    
    return image_coded