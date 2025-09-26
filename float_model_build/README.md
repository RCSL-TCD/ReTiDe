# Pix2Pix U-Net Generator Retrained for Denoising
Cloned repo of pix2pix unet model for hardware adaptation. 

To test model see `my_models.py`

To check out the model see `my_models.py`

Training code is in `train_denoiser.py`

### Git LFS Instructions
```bash
# 📦 Downloading the Model with Git LFS (Linux)

# 🧩 Step 1: Install Git LFS
sudo apt update
sudo apt install git-lfs

# 🔧 Step 2: Initialize Git LFS (one-time setup)
git lfs install

# 📥 Step 3: Clone the repository (automatically fetches LFS-tracked files)
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo

# 🧪 Step 4: Verify the model file downloaded correctly
# This should show a large file size (e.g., 200+ MB), not a tiny text pointer
ls -lh path/to/your_model.pth

# 🛠️ Troubleshooting: If the file is missing or looks like a text pointer, run:
git lfs pull

# ✅ Notes:
# - Git LFS is only required once per system.
# - If Git LFS is not installed, large files will appear as small text pointer files.
# - Do not edit or commit LFS pointer files manually unless you know what you're doing.