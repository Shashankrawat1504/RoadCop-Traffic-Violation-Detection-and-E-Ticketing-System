import torch
from pathlib import Path
import yaml
import os

ROOT = Path(__file__).resolve().parents[1]
dataset_path = str(ROOT / "data" / "datasets")
required_dirs = [
    f"{dataset_path}/train/images",
    f"{dataset_path}/valid/images",
    f"{dataset_path}/test/images"
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        raise RuntimeError(f"Directory not found: {dir_path}")

# Load data configuration
with open(ROOT / 'config' / 'coco128.yaml', 'r') as f:
    data_dict = yaml.safe_load(f)

# Training command
cmd = (
    f'python "{ROOT / "yolov5" / "train.py"}" '
    '--img 640 '
    '--batch 32 '
    '--epochs 5000 '
    f'--data "{ROOT / "config" / "coco128.yaml"}" '
    f'--weights "{ROOT / "models" / "yolov5s.pt"}" '
    '--cache '
    '--device 0 '
    '--patience 30 '
    '--save-period 10 '
    '--rect '
    '--multi-scale '
    '--cos-lr '
    '--label-smoothing 0.1'
)

# Run training
os.system(cmd)

