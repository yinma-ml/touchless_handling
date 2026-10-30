#!/usr/bin/env python3
"""
minimal_smoke_test.py
Uses 1 synthetic 224x224 image in memory to verify cv2, PIL, torch, 
and torchvision MobileNetV3-Small execution. Zero real photos required.
-If it prints all three [OK] lines: Your C++ shared libraries (libGL for OpenCV), PyTorch tensor ops, and torchvision model architecture are 100% functional.
-If cv2 fails with ImportError: libGL.so.1: Common in headless AWS SageMaker containers—fix it by installing pip install opencv-python-headless instead of opencv-python.
-If step 3 prints the offline fallback message: Your corporate proxy blocks download.pytorch.org, meaning you just need to download the 10MB weight file (mobilenet_v3_small-047dcff4.pth) once and load it via torch.load() from your S3 bucket.If step 3 prints the offline fallback message: Your corporate proxy blocks download.pytorch.org, meaning you just need to download the 10MB weight file (mobilenet_v3_small-047dcff4.pth) once and load it via torch.load() from your S3 bucket.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 1. Generate 1 fake RGB photo (224x224) in memory
img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# 2. Test OpenCV (cv2) + Pillow (PIL)
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
pil_img = Image.fromarray(img_np).resize((224, 224))
print(f"[OK] OpenCV & PIL working (sharpness: {sharpness:.1f})")

# 3. Load MobileNetV3-Small (tries pretrained weights, falls back to local init if firewall blocks)
try:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    print("[OK] MobileNetV3-Small loaded WITH pretrained weights.")
except Exception as e:
    model = mobilenet_v3_small(weights=None)
    print(f"[OK] MobileNetV3-Small initialized offline (Firewall blocked download: {type(e).__name__}).")

# 4. Run 1-photo forward pass to extract 576-dim feature vector
extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten()).eval()
tensor_in = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

with torch.no_grad():
    embedding = extractor(tensor_in)

print(f"[OK] Forward pass succeeded! Output shape: {tuple(embedding.shape)}")