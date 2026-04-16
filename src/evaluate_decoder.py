import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from decoder import LightingDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LightingDecoder().to(device)
model.load_state_dict(torch.load("../outputs/checkpoints/lighting_decoder.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

original_image = Image.open("../data/processed/ficus/resized_images/r_0.png").convert("RGB")
original_tensor = transform(original_image).unsqueeze(0).to(device)

lighting_codes = {
    "dark": [0.7, 0.0],
    "orig": [1.0, 0.0],
    "bright": [1.3, 0.0],
    "warm": [1.0, 0.8],
    "cool": [1.0, -0.8],
}

fig, axes = plt.subplots(1, 6, figsize=(18, 3))

axes[0].imshow(original_image)
axes[0].set_title("input")
axes[0].axis("off")

for i, (name, code) in enumerate(lighting_codes.items()):
    lighting_tensor = torch.tensor([code], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(original_tensor, lighting_tensor)

    image = output.squeeze().permute(1, 2, 0).cpu().numpy()

    axes[i + 1].imshow(image)
    axes[i + 1].set_title(name)
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig("../outputs/reconstructions/lighting_grid_v2.png")
plt.show()