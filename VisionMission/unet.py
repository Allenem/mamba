import time
import torch
from torch import nn


class UnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # x: a tensor of shape (batch_size, in_channels, layer_height, layer_width)

        out_before_pooling = self.convs(x)
        out = self.maxpool(out_before_pooling)

        return out, out_before_pooling


# UNetUpBlock: upsampling + concat + Conv + ReLU
class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.upsample = nn.Upsample(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(in_channels * 2,
                                           out_channels,
                                           2,
                                           stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_bridge):
        # x: a tensor of shape (batch_size, in_channels, layer_height // 2, layer_width // 2)
        # x_bridge: a tensor of shape (batch_size, in_channels, layer_height, layer_width)
        x_up = self.upsample(x)
        x_concat = torch.cat([x_up, x_bridge], dim=1)
        out = self.convs(x_concat)

        return out


class Unet(nn.Module):
    def __init__(self, n_base_channels=64):
        super().__init__()

        self.down_blocks = nn.ModuleList([
            UnetDownBlock(3, n_base_channels),
            UnetDownBlock(n_base_channels * 1, n_base_channels * 2),
            UnetDownBlock(n_base_channels * 2, n_base_channels * 4),
            UnetDownBlock(n_base_channels * 4, n_base_channels * 8),
            UnetDownBlock(n_base_channels * 8, n_base_channels * 16)
        ])
        self.up_blocks = nn.ModuleList([
            UnetUpBlock(n_base_channels * 8, n_base_channels * 8),
            UnetUpBlock(n_base_channels * 4, n_base_channels * 4),
            UnetUpBlock(n_base_channels * 2, n_base_channels * 2),
            UnetUpBlock(n_base_channels * 1, n_base_channels * 1),
        ])
        self.final_block = nn.Sequential(
            nn.Conv2d(n_base_channels, 1, kernel_size=1), )

    def forward(self, x):

        out = x
        outputs_before_pooling = []
        for i, block in enumerate(self.down_blocks):
            out, before_pooling = block(out)
            outputs_before_pooling.append(before_pooling)
        out = before_pooling
        # last downblock without pooling
        # now outputs_before_pooling = [block1_before_pooling, ..., block5_before_pooling]

        for i, block in enumerate(self.up_blocks):
            # concat [i, -i-2], eg. [0, -2], [1, -3], [2, -4], [3, -5]
            out = block(out, outputs_before_pooling[-i - 2])
        out = self.final_block(out)

        return out.sigmoid()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    image_size = [256, 256]
    x = torch.Tensor(2, 3, image_size[0], image_size[1]).to(device)
    print(f'input size: {x.size()}')
    
    ts = time.time()
    for i in range(1000):
        y = model(x)
    print(f'Used time: {time.time()-ts:.2f}s')
    print(f'output size: {y.size()}')