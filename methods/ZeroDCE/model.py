import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        r = torch.split(x_r, 3, dim=1)

        for ri in r:
            x = x + ri * (x * x - x)

        return x


class ZeroDCERunner:
    def __init__(self, weight_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = ZeroDCE().to(self.device)
        state_dict = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def run(self, image, save_dir, name):

        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "maps"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "curves"), exist_ok=True)

        input_np = np.array(image).astype(np.float32)
        img = input_np / 255.0

        tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        output_np = output.squeeze().permute(1,2,0).cpu().numpy()
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

        # Save image
        Image.fromarray(output_np).save(os.path.join(save_dir, "images", f"{name}.png"))

        # ------------------------
        # INTERPRETABILITY
        # ------------------------

        # Enhancement map
        enhancement = output_np.astype(float) - input_np.astype(float)

        plt.imshow(enhancement / np.max(np.abs(enhancement)))
        plt.colorbar()
        plt.title("Enhancement Map")
        plt.savefig(os.path.join(save_dir, "maps", f"{name}_enhancement.png"))
        plt.close()

        # Intensity curve
        plt.scatter(input_np.flatten(), output_np.flatten(), s=1)
        plt.xlabel("Input Intensity")
        plt.ylabel("Output Intensity")
        plt.title("Intensity Mapping")
        plt.savefig(os.path.join(save_dir, "curves", f"{name}_curve.png"))
        plt.close()

        return output_np