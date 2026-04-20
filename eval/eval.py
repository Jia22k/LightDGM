import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

loss_fn = lpips.LPIPS(net='alex').to("cuda")

def brightness_map(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def compute_metrics(output_img, gt):
    output_img_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
    gt_img_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)

    print ("GT shape, Output shape:", gt_img_gray.shape, output_img_gray.shape)

    ssim_val = ssim(output_img_gray, gt_img_gray, data_range=255)

    brightness_diff = np.mean(brightness_map(gt)) - np.mean(brightness_map(output_img))

    hist1 = cv2.calcHist([output_img], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([gt], [0], None, [256], [0,256])
    hist_diff = np.linalg.norm(hist1 - hist2)

    lpips_val = loss_fn(
        torch.tensor(output_img/255.).permute(2,0,1).unsqueeze(0).float().to("cuda"),
        torch.tensor(gt/255.).permute(2,0,1).unsqueeze(0).float().to("cuda")
    ).item()

    return {
        "ssim": float(ssim_val),
        "lpips": float(lpips_val),
        "brightness_diff": float(brightness_diff),
        "hist_diff": float(hist_diff)
    }