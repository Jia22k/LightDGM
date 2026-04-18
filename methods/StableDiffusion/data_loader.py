import os
from PIL import Image

class LoLDataset:
    def __init__(self, root_dir):
        self.paths = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith((".png", ".jpg"))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((512, 512))
        return img, path