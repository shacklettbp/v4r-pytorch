import torch
import torchvision
import v4r_example
import sys
import os
import argparse
from timeit import default_timer as timer

if len(sys.argv) != 5:
    print("test.py path/to/stokes.glb GPU_ID BATCH_SIZE MODEL")
    sys.exit(1)

scene_path = sys.argv[1]
gpu_id = int(sys.argv[2])
batch_size = int(sys.argv[3])

device = torch.device(f'cuda:{gpu_id}')

script_dir = os.path.dirname(os.path.realpath(__file__))
views = script_dir + "/stokes_views"

renderer = v4r_example.V4RExample(scene_path, views, gpu_id, batch_size)
tensors = [renderer.getColorTensor(0), renderer.getColorTensor(1)]

print("Initialized and loaded")

num_frames = 30000
num_iters = 30000 // batch_size

start = timer()

sync = renderer.render()

def mlstuff(tensor):
    # Transpose to NCHW
    nchw = tensor.permute(0, 3, 1, 2)

    # Chop off alpha channel
    rgb = nchw[:, 0:3, :, :]

for i in range(1, num_iters):
    nextsync = renderer.render()
    sync.wait()

    mlstuff(tensors[(i - 1) % 2])

    torch.cuda.synchronize()

    sync = nextsync


end = timer()

print("Done")

print(num_iters * batch_size / (end - start))
