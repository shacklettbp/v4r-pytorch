import torch
import torchvision
import v4r_example
import sys
import os
import math
import random
import numpy as np

if len(sys.argv) != 3:
    print("test.py path/to/stokes.glb GPU_ID")
    sys.exit(1)

script_dir = os.path.dirname(os.path.realpath(__file__))
views = script_dir + "/stokes_views"
out_dir = script_dir + "/out"
os.makedirs(out_dir, exist_ok=True)

renderer = v4r_example.V4RExample(sys.argv[1], views, int(sys.argv[2]), 32)

print("Initialized and loaded")

tensors = [renderer.getColorTensor(0), renderer.getColorTensor(1)]

def make_lookat(degrees):
    x = 2 * math.cos(math.radians(degrees))
    z = 2 * math.sin(math.radians(degrees))
    return torch.tensor([[x, 0, z], [0, 0, 0]], dtype=torch.float32).cpu()

lookats = [make_lookat(random.uniform(0, 360)) for i in range(5 * 32)]

for i in range(5):
    print(f"Rendering batch {i}")
    sync = renderer.renderViews(lookats[i*32:i*32+32])
    sync.wait()

    tensor = tensors[i % 2]

    # Single image (grid version)
    grid = tensor.view(4, 8, 256, 256, 4).permute(0, 2, 1, 3, 4)
    grid = grid.reshape(256 * 4, 256 * 8, 4)

    # Transpose to NCHW
    nchw = tensor.permute(0, 3, 1, 2)
    grid = grid.permute(2, 0, 1)

    # Chop off alpha channel
    rgb = nchw[:, 0:3, :, :]
    grid = grid[0:3, :, :]

    img = torchvision.transforms.ToPILImage()(grid.cpu())
    img.save(f"{out_dir}/{i}.png")

    for j in range(32):
        img = torchvision.transforms.ToPILImage()(rgb[j].cpu())
        img.save(f"{out_dir}/{i}_{j}.png")

print("Done")
