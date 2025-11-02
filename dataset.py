import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch # Import torch for testing range

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=128):
        self.hr_dir = hr_dir
        # Find all valid image files in the directory
        self.image_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size), # Apply random crop
            transforms.ToTensor(),             # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Gets a single image item from the dataset by index."""
        img_path = self.image_files[idx]
        # Use 'RGB' mode to ensure 3 channels
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        tensor_image = self.transform(image)
        
        return tensor_image

if __name__ == '__main__':
    # --- Test the dataset loader ---
    
    # Create a fake directory and images for testing
    test_dir = 'data/fake_hr'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Creating fake data in {test_dir}...")
        for i in range(5):
            # Create dummy 512x512 red images
            fake_img = Image.new('RGB', (512, 512), color = 'red')
            fake_img.save(os.path.join(test_dir, f'fake_{i}.png'))
            
    dataset = DIV2KDataset(hr_dir=test_dir, patch_size=128)
    print(f"Dataset size: {len(dataset)}")
    
    # Test with DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get one batch of data
    try:
        first_batch = next(iter(dataloader))
        print(f"Batch data shape: {first_batch.shape}") # Should be (2, 3, 128, 128)
        print(f"Data range: min={first_batch.min():.2f}, max={first_batch.max():.2f}") # Should be around [-1, 1]
    except StopIteration:
        print("Error: DataLoader is empty.")
