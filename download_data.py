import os
import shutil
import kagglehub

# 1. Authenticate with your exact 32-character token (No extra 'e'!)
os.environ["KAGGLE_API_TOKEN"] = "KGAT_4f97552ce6d98f249d1148074910f46"

print("Downloading Alzheimer's dataset from Kaggle...")

# 2. Use the unrestricted dataset that doesn't require website consent
source_path = kagglehub.dataset_download("uraninjo/augmented-alzheimer-mri-dataset-v2")

# 3. Move it from the cache to your project's data folder
dest_path = "data/alzheimers"
if os.path.exists(dest_path):
    shutil.rmtree(dest_path) 

print(f"Moving extracted files to {dest_path}...")
shutil.copytree(source_path, dest_path)

print("SUCCESS! The data is ready in your VS Code folder.")