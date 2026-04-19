import torch
import numpy as np
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import os

class ControlNetRunner:
    def __init__(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

    def get_canny(self, image):
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges = np.stack([edges]*3, axis=-1)
        return Image.fromarray(edges)

    def run(self, image, prompt, negative_prompt, save_dir, name, strength):

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "timesteps"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "metrics"), exist_ok=True)

        timestep_dir = os.path.join(save_dir, "timesteps", name)
        os.makedirs(timestep_dir, exist_ok=True)

        control_image = self.get_canny(image)

        brightness_list = []
        red_list = []
        blue_list = []

        def brightness_map(img):
            return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

        def callback(step, timestep, latents):
            with torch.no_grad():
                decoded = self.pipe.vae.decode(latents / 0.18215).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = decoded.cpu().permute(0,2,3,1).numpy()[0]
                decoded = (decoded * 255).astype(np.uint8)

            Image.fromarray(decoded).save(f"{timestep_dir}/step_{step}.png")

            brightness_list.append(np.mean(brightness_map(decoded)))
            red_list.append(np.mean(decoded[:,:,0]))
            blue_list.append(np.mean(decoded[:,:,2]))

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=control_image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=50,
            callback=callback,
            callback_steps=1
        )

        final_image = output.images[0]
        final_np = np.array(final_image)

        final_path = os.path.join(save_dir, "images", f"{name}.png")
        final_image.save(final_path)

        metrics_dir = os.path.join(save_dir, "metrics")

        np.save(os.path.join(metrics_dir, f"{name}_brightness.npy"), brightness_list)
        np.save(os.path.join(metrics_dir, f"{name}_red.npy"), red_list)
        np.save(os.path.join(metrics_dir, f"{name}_blue.npy"), blue_list)

        return final_np