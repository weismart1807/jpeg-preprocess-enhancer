# JPEG Pre-Processing Enhancer

This project implements a JPEG image enhancement method inspired by the paper from Shoda et al. [2]. The core idea is to pre-process the image before standard JPEG compression. This pre-processing aims to suppress severe artifacts, such as red block noise and pseudo-contours, that occur during low-quality compression (e.g., Q=10).

This project uses a U-Net (`model.py`) as a "color transformation network" and a differentiable JPEG simulator (`utils.py`, based on necla-ml/Diff-JPEG [1]) for end-to-end training.

---

## 1. Installation

Using Conda to manage the environment is recommended.

### Step 1: Clone the Repository

```bash
# Replace with your repository URL
git clone [https://github.com/weismart1807/jpeg-preprocess-enhancer.git](https://github.com/weismart1807/jpeg-preprocess-enhancer.git)
cd jpeg-preprocess-enhancer
