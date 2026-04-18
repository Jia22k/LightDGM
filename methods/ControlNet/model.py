import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# Load controlnet (edge-based)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

model = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load image
image = Image.open("../../data/LoL/our485/low/2.png").convert("RGB")
image = image.resize((512, 512))

# Convert to numpy for edge detection
image_np = np.array(image)

# Canny edges
edges = cv2.Canny(image_np, 100, 200)
edges = np.stack([edges]*3, axis=2)  # make 3-channel
edges = Image.fromarray(edges)

# Prompt
prompt = "same exact scene, brighter lighting, preserve all details"

# Run pipeline
output = model(
    prompt=prompt,
    image=image,              # original image
    control_image=edges,      # edges
    strength=0.5,
    guidance_scale=7.5,
    num_inference_steps=50
)

result = output.images[0]
result.save("controlnet_output.png")