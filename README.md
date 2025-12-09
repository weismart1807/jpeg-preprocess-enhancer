# JPEG Pre-Processing Enhancer

This project implements a JPEG image enhancement method inspired by the paper from Shoda et al. [1]. The core idea is to pre-process the image before standard JPEG compression. This pre-processing aims to suppress severe artifacts, such as red block noise and pseudo-contours, that occur during low-quality compression (e.g., Q=10).

This project uses a U-Net (`model.py`) as a "color transformation network" and a differentiable JPEG simulator (`utils.py`, based on necla-ml/Diff-JPEG [2]) for end-to-end training.

#### This project serves as Digital Image Processing final project, the class is taught by Professor Tang, Chih-Wei.

### 🔍 Project Overview (New and Recommended)

Low-quality JPEG compression introduces visible artifacts such as:
+ Blocking
+ Banding / pseudo-contours
+ Red color noise or shifts

Traditional deep-learning enhancement works operate after decoding, but this requires extra compute on the user’s device.

This project instead enhances images before JPEG compression, training a network that “prepares" the image so that:

+ The JPEG encoder damages it less
+ Its perceptual quality remains high
+ The decoder remains unchanged → standard-compliant

### 🧠 Pipeline Overview
![alt text](Arch.png)

---

## 1. Installation

Using Conda to manage the environment is recommended.

### Step 1: Clone the Repository

```bash
# Replace with your repository URL
git clone https://github.com/weismart1807/jpeg-preprocess-enhancer.git
cd jpeg-preprocess-enhancer
```

### Step 2: Create Conda Environment
We will create a new environment named DIP_env and install the necessary packages.

```bash
#Create the environment
conda create -n DIP_env python=3.9

#Activate the environment
conda activate DIP_env

#Install PyTorch (e.g., for CUDA 12.1+, adjust based on your hardware)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

#Install OpenCV (including contrib modules)
conda install -c conda-forge opencv

#Install other dependencies (DiffJPEG, LPIPS, scikit-image, etc.)
pip install matplotlib scikit-image lpips diff-jpeg 
```

## 2. Dataset Setup
The scripts read data from the `./data/` directory. Since the dataset files are large, they are ignored by .gitignore. You must download them manually and place them according to the following structure:

```bash
jpeg-preprocess-enhancer/ (Your project root)
│
├── data/
│   ├── DIV2K_train_HR/
│   │   └── images/
│   │       ├── 0001.png
│   │       ├── 0002.png
│   │       └── ... (800 training images)
│   │
│   ├── DIV2K_valid_HR/
│   │   └── images/
│   │       ├── 0801.png
│   │       ├── 0802.png
│   │       └── ... (100 validation images)
│   │
│   └── test/
│       ├── 00015_TE_3680x2456.png
│       └── ... (Other test images)
│
├── train.py
├── test.py
├── test_all.py
├── utils.py
├── model.py
├── utils_diff_jpeg_simple.py
├── utils_diff_jpeg_old.py
└── utils_diff_jpeg_new.py
```

### Dataset Links
Training/Validation Data (DIV2K):
[DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)
+ You need to download DIV2K_train_HR.zip (800 images) and DIV2K_valid_HR.zip (100 images).
+ Unzip and place the images into the
```bash
data/DIV2K_train_HR/images #folder
data/DIV2K_valid_HR/images #folder
```

Test Data (JPEG-AI):
[JPEG-AI Test Set](https://jpegai.github.io/test_images/)
+ Download your preferred high-resolution test images (e.g., 00015_TE_3680x2456.png) and place them in
```bash
data/test #folder
```

Note: Ensure you are always in the (DIP_env) environment.

## 3. Training
The train.py script will automatically load the dataset from the data/ directory and begin training. The training parameters (like JPEG_QUALITY = 10) are set within the script, as per the Shoda et al. paper.

### Select Differentiable JPEG Implementation (Line 16)
```bash
# --- Select Differentiable JPEG Implementation ---
from utils_diff_jpeg_new import differentiable_jpeg
# from utils_diff_jpeg_old import differentiable_jpeg
# from utils_diff_jpeg_simple import differentiable_jpeg
```
| Implementation          | File                        | Description                                                                        | Pros                                                                       | Cons                                                       | Recommended Use              |
| ----------------------- | --------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------- |
| **Simple**              | `utils_diff_jpeg_simple.py` | Minimal differentiable JPEG with basic rounding approximation                      | Fast, easy to understand                                                   | Low accuracy, unstable gradients                           | Educational use / debugging  |
| **Old (Approximation)** | `utils_diff_jpeg_old.py`    | Based on Shoda et al. + Shin et al. approximation of rounding                      | Closer to real JPEG, moderate stability                                    | Still inaccurate in quantization behavior; gradient issues | Reproducing older research   |
| **New (SOTA)**          | `utils_diff_jpeg_new.py`    | Based on Reich et al. WACV 2024 “Differentiable JPEG: The Devil Is in the Details” | Best simulator accuracy, stable gradients, realistic quantization modeling | Slightly slower                                            | **Recommended for training** |


### Training command
```bash
python train.py
```
After training, the best model will be saved as best_model.pth. A loss curve plot (loss_plot_q10.png) will also be generated.

## 4. Testing
The test.py script is used to evaluate a single image, comparing the performance of "Standard JPEG" against "Our Enhanced Method" and calculating PSNR, SSIM, and LPIPS metrics.

### Testing command
```bash
python test.py --input "data/test/00015_TE_3680x2456.png" --model "best_model.pth" --quality 10
```
+ --input: (Required) Path to the test image in your ./data/test/ folder.
+ --model: (Required) Path to your trained model weights file (e.g., best_model.pth).
+ --quality: (Required) The JPEG quality to test against (e.g., 10 to match the training).

After running, a Matplotlib window will pop up showing the Original, Standard JPEG, and Our Method side-by-side with their quantitative metrics.

To test all images, update the parameters in `test_all.py` and run:
```bash
python test_all.py 
```

## 5. Result
<img width="1263" height="368" alt="image" src="https://github.com/user-attachments/assets/1aa098e3-f395-4f4f-88f5-d30a02a101bc" />

## 6. References
>[1] Shoda, Akane, Tomo Miyazaki, and Shinichiro Omachi. "JPEG image enhancement with pre-processing of color reduction and smoothing." Sensors 23, no. 21 (2023): 8861.

>[2] Reich, Christoph, Biplob Debnath, Deep Patel, and Srimat Chakradhar. "Differentiable jpeg: The devil is in the details." In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 4126-4135. 2024.
