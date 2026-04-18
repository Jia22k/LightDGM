import torch
import numpy as np
import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import lpips

# ----------------------------
# Setup
# ----------------------------
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("../../data/LoL/our485/low/2.png").convert("RGB")
init_image = init_image.resize((512, 512))

init_np = np.array(init_image)

os.makedirs("timesteps", exist_ok=True)

brightness_list = []
red_list = []
blue_list = []

# ----------------------------
# Helper functions
# ----------------------------

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

    return ssim_val, brightness_diff, hist_diff

# ----------------------------
# CALLBACK (KEY PART)
# ----------------------------

def callback(step, timestep, latents):
    with torch.no_grad():
        image = pipe.vae.decode(latents / 0.18215).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0,2,3,1).numpy()[0]
        image = (image * 255).astype(np.uint8)

    # Save image
    Image.fromarray(image).save(f"timesteps/step_{step}.png")

    # Compute stats
    bright = np.mean(brightness_map(image))
    red = np.mean(image[:,:,0])
    blue = np.mean(image[:,:,2])

    brightness_list.append(bright)
    red_list.append(red)
    blue_list.append(blue)

# ----------------------------
# Run model
# ----------------------------

output = pipe(
    prompt="same scene, bright natural lighting, high exposure, well lit",
    negative_prompt="dark, low light, underexposed, dim lighting, shadows, blue tint, cold lighting",
    image=init_image,
    strength=0.15,
    guidance_scale=7.5,
    num_inference_steps=50,
    callback=callback,
    callback_steps=1
)

final_image = output.images[0]
final_np = np.array(final_image)

final_image.save("output.png")

# ----------------------------
# Metrics (FINAL OUTPUT)
# ----------------------------

ssim_val, brightness_diff, hist_diff = compute_metrics(init_np, final_np)

loss_fn = lpips.LPIPS(net='alex').to("cuda")

lpips_val = loss_fn(
    torch.tensor(init_np/255.).permute(2,0,1).unsqueeze(0).float().to("cuda"),
    torch.tensor(final_np/255.).permute(2,0,1).unsqueeze(0).float().to("cuda")
)

print("LPIPS:", lpips_val.item())

print("SSIM:", ssim_val)
print("Brightness Change:", brightness_diff)
print("Histogram Shift:", hist_diff)

# ----------------------------
# Save stats
# ----------------------------

np.save("brightness.npy", np.array(brightness_list))
np.save("red.npy", np.array(red_list))
np.save("blue.npy", np.array(blue_list))