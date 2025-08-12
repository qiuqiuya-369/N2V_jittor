import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
import PIL.Image as pil_image
import copy
import jittor as jt  # 替换torch为jittor
from jittor.dataset import Dataset as JittorDataset  # Jittor数据集基类

jt.flags.use_cuda = 1  # 启用CUDA

def data_aug(img, mode=0):
    # data augmentation (实现保持不变)
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patch(datasets_path, patch_size, stride):
    # 实现保持不变
    file_list = sorted(glob.glob(datasets_path + '/*'))
    data = []
    patches = []

    for i in range(len(file_list)):
        clean_image = pil_image.open(file_list[i]).convert('RGB')
        for j in range(0, clean_image.width - patch_size + 1, stride):
            for k in range(0, clean_image.height - patch_size + 1, stride):
                x = clean_image.crop((j, k, j + patch_size, k + patch_size))
                patches.append(x)
                for m in range(0, 1):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    for patch in patches:
        data.append(patch)

    return data

class Dataset(JittorDataset):  # 继承JittorDataset
    def __init__(self, images_dir, patch_size, gaussian_noise_level, 
                 downsampling_factor, jpeg_quality, is_gray=True, 
                 batch_size=1, shuffle=False, num_workers=0):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.is_gray = is_gray
        
        # 设置数据集长度 (20倍图像数量)
        self.total_len = len(self.image_files) * 20

    def __len__(self):
        return self.total_len

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def center_pixel_mask(self, img):
        h, w, _ = img.shape
        center_pixel_h = h // 2
        center_pixel_w = w // 2
        random_pixel_h = np.random.randint(0, h)
        random_pixel_w = np.random.randint(0, w)

        while (center_pixel_h == random_pixel_h) and (center_pixel_w == random_pixel_w):
            random_pixel_h = np.random.randint(0, h)
            random_pixel_w = np.random.randint(0, w)

        target = copy.deepcopy(img)
        source = copy.deepcopy(img)
        source[center_pixel_h, center_pixel_w, :] = img[random_pixel_h, random_pixel_w, :]
        return source, target, [center_pixel_h, center_pixel_w]

    def __getitem__(self, idx):
        idx = idx % len(self.image_files)
        if self.is_gray:
            clean_image1 = pil_image.open(self.image_files[idx]).convert('L')
        else:
            clean_image1 = pil_image.open(self.image_files[idx]).convert('RGB')

        crop_x = random.randint(0, clean_image1.width - self.patch_size)
        crop_y = random.randint(0, clean_image1.height - self.patch_size)
        clean_image1 = clean_image1.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        noisy_image1 = clean_image1.copy()

        # 高斯噪声处理
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])

            if self.is_gray:
                gaussian_noise1 = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size)).astype(np.float32)
            else:
                gaussian_noise1 = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size, 3)).astype(np.float32)

        noisy_image1 = np.array(noisy_image1).astype(np.float32)
        noisy_image1 += gaussian_noise1

        if self.is_gray:
            noisy_image1 = np.expand_dims(noisy_image1, axis=-1)
            source, target, blind_pos = self.center_pixel_mask(noisy_image1)
        else:
            source, target, blind_pos = self.center_pixel_mask(noisy_image1)

        source, target = self.random_horizontal_flip(source, target)
        source, target = self.random_vertical_flip(source, target)
        source, target = self.random_rotate_90(source, target)

        # 通道调整和归一化
        if self.is_gray:
            input = np.transpose(source, axes=[2, 0, 1])
            label = np.transpose(target, axes=[2, 0, 1])
        else:
            input = np.transpose(source, axes=[2, 0, 1])
            label = np.transpose(target, axes=[2, 0, 1])

        input /= 255.0
        label /= 255.0

        # 转换为Jittor数组
        input = jt.array(input, dtype=jt.float32)
        label = jt.array(label, dtype=jt.float32)
        blind_pos = jt.array(blind_pos)

        return input, label, blind_pos

class EvalDataset(JittorDataset):  # 继承JittorDataset
    def __init__(self, clean_image_dir, gaussian_noise_level, 
                 downsampling_factor, jpeg_quality, is_gray=False,
                 batch_size=1, shuffle=False, num_workers=0):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.clean_image_files = sorted(glob.glob(clean_image_dir + '/*'))
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.is_gray = is_gray
        self.total_len = len(self.clean_image_files)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.is_gray:
            clean_image = pil_image.open(self.clean_image_files[idx]).convert('L')
        else:
            clean_image = pil_image.open(self.clean_image_files[idx]).convert('RGB')

        noisy_image1 = clean_image.copy()

        # 高斯噪声处理
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
                
            if self.is_gray:
                gaussian_noise1 = np.zeros((clean_image.height, clean_image.width), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width)).astype(np.float32)
            else:
                gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(np.float32)

        # 下采样和JPEG处理（原代码注释保留）
        # ...

        noisy_image1 = np.array(noisy_image1).astype(np.float32)
        clean_image = np.array(clean_image).astype(np.float32)
        noisy_image1 += gaussian_noise1

        if self.is_gray:
            noisy_image1 = np.expand_dims(noisy_image1, axis=-1)
            clean_image = np.expand_dims(clean_image, axis=-1)
            input = np.transpose(noisy_image1, axes=[2, 0, 1])
            label = np.transpose(clean_image, axes=[2, 0, 1])
        else:
            input = np.transpose(noisy_image1, axes=[2, 0, 1])
            label = np.transpose(clean_image, axes=[2, 0, 1])

        input /= 255.0
        label /= 255.0

        # 转换为Jittor数组
        input = jt.array(input, dtype=jt.float32)
        label = jt.array(label, dtype=jt.float32)

        return input, label