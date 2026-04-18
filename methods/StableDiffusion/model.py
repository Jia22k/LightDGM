import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

class StableDiffusionRunner:
    def __init__(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")

    def run(self, image, prompt, negative_prompt, save_dir, name, strength=0.3):

        os.makedirs(save_dir, exist_ok=True)
        timestep_dir = os.path.join(save_dir, "timesteps", name)
        os.makedirs(timestep_dir, exist_ok=True)

        brightness_list = []
        red_list = []
        blue_list = []

        def brightness_map(img):
            return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

        def callback(step, timestep, latents):
            with torch.no_grad():
                image_decoded = self.pipe.vae.decode(latents / 0.18215).sample
                image_decoded = (image_decoded / 2 + 0.5).clamp(0, 1)
                image_decoded = image_decoded.cpu().permute(0,2,3,1).numpy()[0]
                image_decoded = (image_decoded * 255).astype(np.uint8)

            Image.fromarray(image_decoded).save(f"{timestep_dir}/step_{step}.png")

            brightness_list.append(np.mean(brightness_map(image_decoded)))
            red_list.append(np.mean(image_decoded[:,:,0]))
            blue_list.append(np.mean(image_decoded[:,:,2]))

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=50,
            callback=callback,
            callback_steps=1
        )

        final_image = output.images[0]
        final_np = np.array(final_image)

        final_path = os.path.join(save_dir, "images", f"{name}.png")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        final_image.save(final_path)

        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        np.save(os.path.join(metrics_dir, f"{name}_brightness.npy"), brightness_list)
        np.save(os.path.join(metrics_dir, f"{name}_red.npy"), red_list)
        np.save(os.path.join(metrics_dir, f"{name}_blue.npy"), blue_list)

        return final_np