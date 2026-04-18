import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

rows = []

# -----------------------------
# PIX2PIX EVALUATION
# -----------------------------
pix2pix_folder = "results/facades_label2photo_pretrained/test_latest/images"

files = sorted(os.listdir(pix2pix_folder))

fake_files = [f for f in files if "fake_B" in f]

for fake_file in fake_files:
    base_name = fake_file.replace("_fake_B.png", "")

    fake_path = os.path.join(pix2pix_folder, fake_file)
    real_path = os.path.join(pix2pix_folder, base_name + "_real_B.png")

    if not os.path.exists(real_path):
        continue

    fake_img = cv2.imread(fake_path)
    real_img = cv2.imread(real_path)

    fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)

    brightness_diff = abs(float(fake_gray.mean()) - float(real_gray.mean()))

    fake_hist = cv2.calcHist([fake_img], [0], None, [256], [0, 256])
    real_hist = cv2.calcHist([real_img], [0], None, [256], [0, 256])
    hist_diff = np.linalg.norm(fake_hist - real_hist)

    ssim_score = ssim(fake_gray, real_gray)

    rows.append({
        "model": "pix2pix",
        "image": base_name,
        "brightness_diff": brightness_diff,
        "histogram_diff": float(hist_diff),
        "ssim": ssim_score
    })

# -----------------------------
# CYCLEGAN EVALUATION
# -----------------------------
cyclegan_folder = "results/horse2zebra_pretrained/test_latest/images"

files = sorted(os.listdir(cyclegan_folder))

fake_files = [f for f in files if "fake" in f]

for fake_file in fake_files:
    fake_path = os.path.join(cyclegan_folder, fake_file)

    img = cv2.imread(fake_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())

    blue_mean = float(img[:, :, 0].mean())
    green_mean = float(img[:, :, 1].mean())
    red_mean = float(img[:, :, 2].mean())

    red_blue_ratio = red_mean / (blue_mean + 1e-5)

    rows.append({
        "model": "cyclegan",
        "image": fake_file,
        "brightness_diff": brightness,
        "histogram_diff": red_blue_ratio,
        "ssim": np.nan
    })

# Save CSV
results_df = pd.DataFrame(rows)
results_df.to_csv("evaluation_results.csv", index=False)

print("Saved evaluation_results.csv")
print(results_df.head())