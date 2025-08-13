import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

# 计算PSNR
def calc_psnr(img1, img2):
    if hasattr(img1, 'dtype') and img1.dtype == np.float32:
        mse = np.mean((img1 - img2) ** 2)
        return 10 * np.log10(1.0 / mse) if mse > 0 else 100
    else:
        return 10 * np.log10(1.0 / np.mean((img1 - img2) ** 2))

# 计算SSIM
def calculate_ssim(img, img2):
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

# 指标计算
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

# 调整学习率
def adjust_lr(optimizer, lr, step, decay_steps, decay_gamma):
    current_lr = lr * (decay_gamma ** ((step + 1) // decay_steps))
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return current_lr
