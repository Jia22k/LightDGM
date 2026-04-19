import os
import json
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.data_loader import LoLDataset
from eval.eval import compute_metrics
from model import ZeroDCERunner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

dataset = LoLDataset(os.path.join(ROOT, "datasets/LoL/eval15/low/"))

runner = ZeroDCERunner(os.path.join(ROOT, "pretrained/zerodce.pth"))

save_root = os.path.join(ROOT, "methods/ZeroDCE/outputs")
os.makedirs(save_root, exist_ok=True)

all_metrics = []

for idx in range(len(dataset)):
    img, path = dataset[idx]
    img_np = np.array(img)

    name = os.path.splitext(os.path.basename(path))[0]

    print(f"Running Zero-DCE: {name}")

    output_np = runner.run(img, save_root, name)

    metrics = compute_metrics(img_np, output_np)
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["image"] = name

    all_metrics.append(metrics)

with open(os.path.join(save_root, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=4)

print("Done Zero-DCE run.")