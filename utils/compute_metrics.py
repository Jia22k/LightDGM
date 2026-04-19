import os
import json
import numpy as np
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from eval.eval import compute_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

input_dir = os.path.join(ROOT, "datasets/LoL/eval15/low/")
gt_dir = os.path.join(ROOT, "datasets/LoL/eval15/high/")
output_dir = os.path.join(ROOT, "methods/EnlightenGAN/outputs/images/")

print (f"Input dir: {input_dir}")

all_metrics = []

for fname in os.listdir(output_dir):
    input_path = os.path.join(input_dir, fname)
    gt_path = os.path.join(gt_dir, fname)
    output_path = os.path.join(output_dir, fname)

    input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
    output_img = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)

    metrics = compute_metrics(gt_img, output_img)
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["image"] = fname

    all_metrics.append(metrics)

with open(os.path.join(ROOT, "methods/EnlightenGAN/outputs/metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=4)