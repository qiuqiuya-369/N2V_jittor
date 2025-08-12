import argparse
import os, glob
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
from model import Unet
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
import pandas as pd
from utils import calc_psnr, calculate_ssim

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='N2V_Unet')
    parser.add_argument('--weights_path', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/weights_50/best_N2V_Unet_[50].pth')  # 最优模型文件
    parser.add_argument('--is_gray', type=str, default='True')  # 是否是灰度图
    parser.add_argument('--images_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/datasets/BSD68_test')    # 测试集文件夹
    parser.add_argument('--outputs_denoising_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/data/BSD68_denoising_50_N2V_Unet')    # 去噪结果保存文件夹
    parser.add_argument('--outputs_plt_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/data/BSD68_denoising_50_plt_N2V_Unet')  # 去噪结果可视化保存文件夹
    parser.add_argument('--gaussian_noise_level', type=str, default='50')   # 高斯噪声强度
    parser.add_argument('--jpeg_quality', type=int)
    parser.add_argument('--downsampling_factor', type=int)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_denoising_dir):
        os.makedirs(opt.outputs_denoising_dir)

    if not os.path.exists(opt.outputs_plt_dir):
        os.makedirs(opt.outputs_plt_dir)

    # 创建CSV文件路径
    csv_path = os.path.join('autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master', 'denoising_metrics.csv')
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if opt.arch == 'N2V_Unet':
        if opt.is_gray:
            model = Unet(1,1)
        else:
            model = Unet(3,3)

    # 读取最优模型
    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    if opt.gaussian_noise_level is not None:
        opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))
        if len(opt.gaussian_noise_level) == 1:
            sigma = opt.gaussian_noise_level[0]
        else:
            sigma = random.randint(opt.gaussian_noise_level[0], opt.gaussian_noise_level[1])

    image_list = glob.glob(opt.images_dir + "/*.*")

    benchmark_len = len(image_list)

    sum_psnr = 0.0
    sum_ssim = 0.0

    # 新增：创建列表存储每张图片的评估结果
    results_data = []

    for i in range(benchmark_len):
        filename = os.path.basename(image_list[i]).split('.')[0]
        descriptions = ''
        print("image:", filename)

        if opt.is_gray:
            input = pil_image.open(image_list[i]).convert('L')
        else:
            input = pil_image.open(image_list[i]).convert('RGB')
        GT = input
        GT_cal = np.array(input).astype(np.float32) / 255.0


        if opt.gaussian_noise_level is not None:
            if opt.is_gray:
                # 加对应噪声水平的噪声
                noise = np.random.normal(0.0, sigma, (input.height, input.width)).astype(np.float32)
                input = np.array(input).astype(np.float32) + noise
                input = np.expand_dims(input, axis=-1)
            else:
            # 加对应噪声水平的噪声
                noise = np.random.normal(0.0, sigma, (input.height, input.width, 3)).astype(np.float32)
                input = np.array(input).astype(np.float32) + noise

            input /= 255.0
            noisy_input = input

        if opt.jpeg_quality is not None:
            buffer = io.BytesIO()
            input.save(buffer, format='jpeg', quality=opt.jpeg_quality)
            input = pil_image.open(buffer)
            descriptions += '_jpeg_q{}'.format(opt.jpeg_quality)
            input.save(os.path.join(opt.outputs_denoising_dir, '{}{}.png'.format(filename, descriptions)))
            input = np.array(input).astype(np.float32)
            input /= 255.0
            noisy_input = input

        if opt.downsampling_factor is not None:
            original_width = input.width
            original_height = input.height
            input = input.resize((input.width // opt.downsampling_factor,
                                  input.height // opt.downsampling_factor),
                                 resample=pil_image.BICUBIC)
            input = input.resize((original_width, original_height), resample=pil_image.BICUBIC)
            descriptions += '_sr_s{}'.format(opt.downsampling_factor)
            input.save(os.path.join(opt.outputs_denoising_dir, '{}{}.png'.format(filename, descriptions)))
            input = np.array(input).astype(np.float32)
            input /= 255.0
            noisy_input = input

        input = transforms.ToTensor()(input).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(input)

        if opt.is_gray:
            output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).squeeze(0).byte().cpu().numpy()
            denoising_output = output / 255.0
            output = pil_image.fromarray(output, mode='L')
        else:
            output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            denoising_output = output / 255.0
            output = pil_image.fromarray(output, mode='RGB')
        output.save(os.path.join(opt.outputs_denoising_dir, '{}_{}_{}.png'.format(filename, descriptions, opt.arch)))

        psnr = calc_psnr(GT_cal, denoising_output)
        ssim = calculate_ssim(GT_cal * 255, denoising_output * 255)
        sum_psnr += psnr
        sum_ssim += ssim
        noisy_input_squeezed = np.squeeze(noisy_input, axis=-1) if noisy_input.ndim == 3 and noisy_input.shape[
            -1] == 1 else noisy_input
        noisy_psnr = calc_psnr(GT_cal, noisy_input_squeezed)

        # 新增：将当前图片结果添加到列表
        results_data.append({
            'filename': filename,
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6)
        })
        print(f"  PSNR: {psnr:.4f} dB, SSIM: {ssim:.6f}")

        # 对比图
        fig, axes = plt.subplots(1, 3)
        # 关闭坐标轴
        for ax in axes:
            ax.axis('off')
        # 在每个子图中显示对应的图像
        if opt.is_gray:
            axes[0].imshow(GT, cmap='gray')
            axes[1].imshow(noisy_input, cmap='gray')
            axes[2].imshow(output, cmap='gray')
        else:
            axes[0].imshow(GT)
            axes[1].imshow(noisy_input)
            axes[2].imshow(output)

        axes[0].set_title('Ground-Truth',fontsize=8)
        axes[1].set_title('Noisy_{} (PSNR: {:.2f})'.format(opt.gaussian_noise_level, noisy_psnr),fontsize=8)
        axes[2].set_title('{}(PSNR: {:.2f})'.format(opt.arch,psnr),fontsize=8)

        # 保存图像
        plt.savefig(
            os.path.join(opt.outputs_plt_dir, '{}_plt_x{}_{}.png'.format(filename, opt.gaussian_noise_level, opt.arch)),
            bbox_inches='tight', dpi=600)
        plt.close()


    # 计算平均指标
    avg_psnr = sum_psnr / benchmark_len
    avg_ssim = sum_ssim / benchmark_len
    print('PSNR: {:.2f}'.format(avg_psnr))
    print('SSIM: {:.4f}'.format(avg_ssim))
    
    # 新增：添加平均值到结果列表（用空行分隔区分）
    results_data.append({
        'filename': 'Average',
        'psnr': round(avg_psnr, 4),
        'ssim': round(avg_ssim, 6)
    })

    # 新增：将结果转换为DataFrame并保存为CSV
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"评估结果已保存至CSV文件: {csv_path}")
