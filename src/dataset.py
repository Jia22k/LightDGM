""" dataset.py

This file is what feeds data into the model during training.

It tells PyTorch:

where images are stored
how to load them
what inputs and targets should be

Right now your dataset returns:

original_image
lighting_tensor
target_image

Example:

input: original Lego image
lighting: [1.3, 0.0]
target: bright Lego image

So the model learns:

"If I see this object and this lighting code,
what should the final image look like?" """

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LightingDataset(Dataset):
    def __init__(self):
        self.original_dir = "../data/processed/ficus/resized_images"
        self.image_dir = "../data/processed/ficus/lighting_variants"
        self.lighting_code_path = "../data/processed/ficus/lighting_codes/lighting_codes.json"

        with open(self.lighting_code_path, "r") as f:
            self.lighting_codes = json.load(f)

        self.image_files = sorted(list(self.lighting_codes.keys()))

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]

        target_path = os.path.join(self.image_dir, filename)
        target_image = Image.open(target_path).convert("RGB")

        base_name = "_".join(filename.split("_")[:2]) + ".png"
        original_path = os.path.join(self.original_dir, base_name)
        original_image = Image.open(original_path).convert("RGB")
        original_image = self.transform(original_image)
        target_image = self.transform(target_image)
        lighting = list(self.lighting_codes[filename].values())
        lighting_tensor = torch.tensor(lighting, dtype=torch.float32)
        return original_image, lighting_tensor, target_image