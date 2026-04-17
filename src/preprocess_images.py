
# This file creates fake lighting conditions.

# For every original image, it  creates:

# dark version
#bright version
#warm version
#cool version

#So one original image becomes many training examples.
from pathlib import Path
from PIL import Image

input_dir = Path("../data/nerf-pytorch/data/nerf_synthetic/lego/train")
output_dir = Path("../data/processed/ficus/resized_images")

output_dir.mkdir(parents=True, exist_ok=True)

image_paths = list(input_dir.glob("*.png"))

print(f"Found {len(image_paths)} images")

for image_path in image_paths:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))

    output_path = output_dir / image_path.name
    img.save(output_path)

print(f"Saved {len(image_paths)} resized images to {output_dir}")