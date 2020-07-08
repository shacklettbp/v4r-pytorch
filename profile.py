import torch
import torchvision
import v4r_example
import sys
import os
import argparse
from timeit import default_timer as timer
import random
import math
import torch.nn as nn

dim = (256, 256)
fp16 = True

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
        use_batch_norm=False,
    ):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.BatchNorm2d(planes) if use_batch_norm else nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.BatchNorm2d(planes) if use_batch_norm else nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def _impl(self, x):
        residual = x
        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

    def forward(self, x):
        return self._impl(x)


def _build_bottleneck_branch(inplanes, planes, ngroups, stride, expansion, groups=1):
    return nn.Sequential(
        conv1x1(inplanes, planes),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion),
        nn.GroupNorm(ngroups, planes * expansion),
    )


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
        use_batch_norm=False,
    ):
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes, planes, ngroups, stride, self.expansion, groups=cardinality,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

    def forward(self, x):
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        cardinality=1,
        use_batch_norm=False,
    ):
        super().__init__(inplanes, planes, ngroups, stride, downsample, cardinality)

        self.se = _build_se_branch(planes * self.expansion)

    def _impl(self, x):
        identity = x

        out = self.convs(x)
        out = self.se(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


class SpaceToDepth(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * 16, H // 4, W // 4)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        base_planes,
        ngroups,
        block,
        layers,
        cardinality=1,
        use_batch_norm=False,
    ):
        super(ResNet, self).__init__()
        self.stem = nn.Sequential(
            torch.jit.script(SpaceToDepth()),
            nn.Conv2d(in_channels * 16, base_planes, kernel_size=1, bias=False),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )

        self.cardinality = cardinality
        self.use_batch_norm = use_batch_norm

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(
            block, ngroups, base_planes * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, ngroups, base_planes * 2 * 2, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2
        )

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

        self.linear = nn.Linear(256 * (dim[0] // 32) * (dim[1] // 32), 1)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
                if self.use_batch_norm
                else nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                cardinality=self.cardinality,
                use_batch_norm=self.use_batch_norm,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, ngroups, use_batch_norm=self.use_batch_norm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def resnet18(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        BasicBlock,
        [2, 2, 2, 2],
    )

    return model


def resnet50(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3])

    return model


def resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resnet50(in_channels, base_planes, ngroups):
    model = ResNet(in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3])

    return model


def se_resneXt50(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_resneXt101(in_channels, base_planes, ngroups):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model

if len(sys.argv) != 6:
    print("test.py path/to/stokes.glb GPU_ID BATCH_SIZE RENDER_MODE ML_MODE")
    sys.exit(1)

num_frames = 30000

scene_path = sys.argv[1]
gpu_id = int(sys.argv[2])
batch_size = int(sys.argv[3])

num_iters = 30000 // batch_size

device = torch.device(f'cuda:{gpu_id}')

script_dir = os.path.dirname(os.path.realpath(__file__))
views = script_dir + "/stokes_views"

if sys.argv[4] == "random":
    tensors = [torch.rand(batch_size, dim[0], dim[1], 4).to(device),
               torch.rand(batch_size, dim[0], dim[1], 4).to(device)]

    def render(cur_iter):
        class FakeSync:
            def wait(self):
                pass

        torch.rand(batch_size, dim[0], dim[1], 4, out=tensors[cur_iter % 2])

        return FakeSync()
else:
    if sys.argv[4] == "rotate":
        coordinate_txfm = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float32).cpu()

        def make_lookat(degrees):
            x = 2 * math.cos(math.radians(degrees))
            z = 2 * math.sin(math.radians(degrees))
            return torch.tensor([[x, 0, z], [0, 0, 0]], dtype=torch.float32).cpu()
        
        lookats = [make_lookat(random.uniform(0, 360)) for i in range(num_iters * batch_size)]

        def render(cur_iter):
            return renderer.renderViews(lookats[cur_iter*batch_size:cur_iter*batch_size+batch_size])

    else:
        coordinate_txfm = torch.tensor([
                [1, 0, 0, 0],
                [0, -1.19209e-07, -1, 0],
                [0, 1, -1.19209e-07, 0],
                [0, 0, 0, 1]], dtype=torch.float32).cpu()

        def render(cur_iter):
            return renderer.render()

    renderer = v4r_example.V4RExample(scene_path, views, gpu_id, batch_size, coordinate_txfm)
    tensors = [renderer.getColorTensor(0), renderer.getColorTensor(1)]

print("Initialized and loaded")

if sys.argv[5] == "resnet18":
    model = resnet18(3, 32, 16)
    if fp16:
        model = model.half()

    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, nn.init.calculate_gain("relu")
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    model = model.to(device)

    model.apply(init_weights)
    model.eval()

    def network(rgb):
        with torch.no_grad():
            return model(rgb)

elif sys.argv[5] == "simple":
    model = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=(3, 4),
        ),
        nn.ReLU(True),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=(1, 1),
        ),
        nn.ReLU(True),
        nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        ),
          nn.ReLU(True),
        Flatten(),
        nn.Linear(32 * (dim[0]//8) * (dim[1]//8), 1),
        nn.ReLU(True)
    )

    if fp16:
        model = model.half()
    model = model.to(device)

    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                layer.weight, nn.init.calculate_gain("relu")
            )
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

    model.eval()

    def network(rgb):
        with torch.no_grad():
            return model(rgb)

else:
    def network(rgb):
        return rgb[:, 0, 0, 0:1]

start = timer()

sync = render(0)

total = 0
for i in range(1, num_iters):
    nextsync = render(i)
    sync.wait()

    tensor = tensors[(i - 1) % 2]

    # Transpose to NCHW
    nchw = tensor.permute(0, 3, 1, 2)

    # Chop off alpha channel
    rgb = nchw[:, 0:3, :, :]

    if fp16:
        r = network(rgb.half().contiguous())
    else:
        r = network(rgb.float().contiguous())
    total += r.mean().cpu().item()

    sync = nextsync


end = timer()

print("Done")
print("Debug: ", total)
print("Render MS per Batch")
print("Network MS per Batch")
print("Aggregate FPS:", num_iters * batch_size / (end - start))

