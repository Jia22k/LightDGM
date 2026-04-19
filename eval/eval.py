import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

loss_fn = lpips.LPIPS(net='alex').to("cuda")

def brightness_map(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def compute_metrics(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    ssim_val = ssim(img1_gray, img2_gray)

    brightness_diff = np.mean(brightness_map(img2)) - np.mean(brightness_map(img1))

    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    hist_diff = np.linalg.norm(hist1 - hist2)

    lpips_val = loss_fn(
        torch.tensor(img1/255.).permute(2,0,1).unsqueeze(0).float().to("cuda"),
        torch.tensor(img2/255.).permute(2,0,1).unsqueeze(0).float().to("cuda")
    ).item()

    return {
        "ssim": float(ssim_val),
        "lpips": float(lpips_val),
        "brightness_diff": float(brightness_diff),
        "hist_diff": float(hist_diff)
    }