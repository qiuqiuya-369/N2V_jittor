
import jittor as jt
from jittor import nn
import math

def pixel_mse_loss(predictions, targets, pixel_pos):
    mask = jt.zeros(targets.shape)
    for i in range(pixel_pos.shape[0]):
        h, w = pixel_pos[i]
        mask[i, :, h, w] = 1.0
    return nn.mse_loss(predictions * mask, targets * mask) * 100000

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Pool(2, op='maximum')
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.Pool(2, op='maximum')
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(96+in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Leaky_relu(0.1)
        )

    def execute(self, x):
        pool1 = self.block1(x)
        pool2 = self.block2(pool1)
        pool3 = self.block2(pool2)
        pool4 = self.block2(pool3)
        pool5 = self.block2(pool4)

        upsample5 = self.block3(pool5)
        concat5 = jt.concat([upsample5, pool4], dim=1)
        upsample4 = self.block4(concat5)
        concat4 = jt.concat([upsample4, pool3], dim=1)
        upsample3 = self.block5(concat4)
        concat3 = jt.concat([upsample3, pool2], dim=1)
        upsample2 = self.block5(concat3)
        concat2 = jt.concat([upsample2, pool1], dim=1)
        upsample1 = self.block5(concat2)
        concat1 = jt.concat([upsample1, x], dim=1)
        output = self.block6(concat1)
        return output