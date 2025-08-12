# import copy
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from PIL import Image, ImageChops
# import cv2

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def draw_line(viz, X, Y, win, linename):
#     viz.line(Y=Y,
#              X=X,
#              win=win,
#              update='append',
#              name=linename)


# def loadIMG_crop(img_path, scale):
#     """load img and Crop to an integer multiple of scale"""
#     hrIMG = Image.open(img_path)
#     lr_wid = hrIMG.width // scale
#     lr_hei = hrIMG.height // scale
#     hr_wid = lr_wid * scale
#     hr_hei = lr_hei * scale
#     hrIMG = hrIMG.crop((0, 0, hr_wid, hr_hei))
#     return hrIMG


# def img2ycbcr(hrIMG, gray2rgb=False):
#     """if gray and not convert it to rgb, return 1 channel"""
#     if not gray2rgb and hrIMG.mode == 'L':
#         hr = np.array(hrIMG)
#         hr = hr.astype(np.float32)
#     else:
#         hrIMG.convert('RGB')
#         hr = np.array(hrIMG)  # whc -> hwc
#         hr = rgb2ycbcr(hr).astype(np.float32)
#     return hr


def calc_psnr(img1, img2):
    if hasattr(img1, 'dtype') and img1.dtype == np.float32:
        mse = np.mean((img1 - img2) ** 2)
        return 10 * np.log10(1.0 / mse) if mse > 0 else 100
    else:
        return 10 * np.log10(1.0 / np.mean((img1 - img2) ** 2))


def calculate_ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    # 确保输入是numpy数组
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2)
    
    # 确保数据类型为float64
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 设置SSIM计算参数
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    # 计算均值
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    # 计算SSIM映射
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def preprocess(img, device, image_mode='RGB'):
#     if image_mode == 'RGB':
#         img = np.array(img).astype(np.float32)  # (width, height, channel) -> (height, width, channel)
#         ycbcr = rgb2ycbcr(img).astype(np.float32).transpose([2, 0, 1])
#         ycbcr /= 255.
#         ycbcr = torch.from_numpy(ycbcr).to(device).unsqueeze(0)  # numpy -> cpu tensor -> GPU tensor
#         y = ycbcr[0, 0, ...].unsqueeze(0).unsqueeze(0)  # input Tensor Dimension: batch_size * channel * H * W
#         return y, ycbcr
#     else:
#         y = img.astype(np.float32)  # (width, height, channel) -> (height, width, channel)
#         y /= 255.
#         y = torch.from_numpy(y).to(device)  # numpy -> cpu tensor -> GPU tensor
#         y = y.unsqueeze(0).unsqueeze(0)  # input Tensor Dimension: batch_size * channel * H * W
#         return y, y


# # helper function for visualizing the output of a given layer
# # default number of filters is 4
# def viz_layer(layer, n_filters=4):
#     plt.figure(figsize=(4, 3.5))
#     min = torch.min(layer).item()
#     max = torch.max(layer).item()
#     # mean = torch.mean(layer).item()
#     # std = torch.std(layer).item()
#     # transforms1 = transforms.Normalize(mean=mean, std=std)
#     for index, filter in enumerate(layer):
#         plt.subplots_adjust(wspace=0.05, hspace=0.05)
#         plt.subplot(8, 8, index + 1)
#         # min = torch.min(filter).item()
#         # max = torch.max(filter).item()
#         filter = (filter - min) / (max - min)
#         # plt.imshow(transforms1(filter.detach())[0, :, :],  cmap='gray')
#         plt.imshow(filter[0, :, :].detach(), cmap='gray')
#         plt.axis('off')
#     plt.show()


# def viz_layer2(layer, n_filters=4):
#     plt.figure(figsize=(4, 3.5))
#     layer = torch.from_numpy(layer)
#     min = torch.min(layer).item()
#     max = torch.max(layer).item()
#     # mean = torch.mean(layer).item()
#     # std = torch.std(layer).item()
#     # transforms1 = transforms.Normalize(mean=mean, std=std)
#     for index in range(n_filters):
#         filter = layer[:, :, index]
#         plt.subplots_adjust(wspace=0.05, hspace=0.05)
#         plt.subplot(8, 8, index + 1)
#         # filter = (filter - min)/(max - min)
#         # plt.imshow(transforms1(filter.detach())[0, :, :],  cmap='gray')
#         plt.imshow(filter[:, :].detach(), cmap='gray')
#         plt.axis('off')
#     plt.show()


def rgb2ycbcr(rgb):
    """RGB转YCbCr"""
    if len(rgb.shape) == 2:  # 灰度图
        return rgb
    elif len(rgb.shape) == 3:
        # 转换矩阵
        m = np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112.0],
                      [112.0, -93.786, -18.214]])
        shape = rgb.shape
        if len(shape) == 3:
            rgb = rgb.reshape((shape[0] * shape[1], 3))
        ycbcr = np.dot(rgb, m.transpose() / 255.8)
        ycbcr[:, 0] += 16.
        ycbcr[:, 1:] += 128.
        ycbcr = np.round(ycbcr)
        return ycbcr.reshape(shape)
    else:
        return rgb


# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    """YCbCr转RGB"""
    if len(ycbcr.shape) == 2:  # 灰度图
        return ycbcr
    elif len(ycbcr.shape) == 3:
        m = np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112],
                      [112, -93.786, -18.214]])
        shape = ycbcr.shape
        if len(shape) == 3:
            ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
        rgb = np.copy(ycbcr)
        rgb[:, 0] -= 16.
        rgb[:, 1:] -= 128.
        rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
        rgb = np.round(rgb)
        return np.clip(rgb, 0, 255).reshape(shape)
    else:
        return ycbcr


# 平移
def ImgOffSet(Img, xoff, yoff):
    width, height = Img.size
    c = ImageChops.offset(Img, xoff, yoff)
    c.paste((0, 0, 0), (0, 0, xoff, height))
    c.paste((0, 0, 0), (0, 0, width, yoff))
    return c

# 调整学习率
# decay_steps：多少个epoch后衰减
# decay_gamma：衰减多少倍
# 训练默认为200个epoch降一半
def adjust_lr(optimizer, lr, step, decay_steps, decay_gamma):
    current_lr = lr * (decay_gamma ** ((step + 1) // decay_steps))
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return current_lr