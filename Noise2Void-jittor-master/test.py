import argparse
import os, glob
import io
import random
from jittor import transform
import numpy as np
import jittor as jt
import pandas as pd
import PIL.Image as pil_image
from model import Unet
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import calc_psnr, calculate_ssim

jt.flags.use_cuda = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='N2V_Unet')
    parser.add_argument('--weights_path', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/weights_50/best_N2V_Unet_[50].pth')  # 最优模型文件
    parser.add_argument('--is_gray', type=str, default='True')  # 是否是灰度图
    parser.add_argument('--images_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/datasets/BSD68_test')    # 测试集文件夹
    parser.add_argument('--outputs_denoising_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/data/BSD68_denoising_50_N2V_Unet')    # 去噪结果保存文件夹
    parser.add_argument('--outputs_plt_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/data/BSD68_denoising_50_plt_N2V_Unet')  # 去噪结果可视化保存文件夹
    parser.add_argument('--gaussian_noise_level', type=str, default='50')   # 高斯噪声强度
    parser.add_argument('--jpeg_quality', type=int) # JPEG压缩质量
    parser.add_argument('--downsampling_factor', type=int) # 下采样因子
    opt = parser.parse_args() # 解析参数并保存到opt对象

    if not os.path.exists(opt.outputs_denoising_dir):
        os.makedirs(opt.outputs_denoising_dir)

    if not os.path.exists(opt.outputs_plt_dir):
        os.makedirs(opt.outputs_plt_dir)

    csv_path = os.path.join('autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master', 'denoising_metrics_50.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if opt.arch == 'N2V_Unet':
        if opt.is_gray:
            model = Unet(1,1)
        else:
            model = Unet(3,3)

    # 加载预训练权重
    state_dict = model.state_dict()
    for n, p in jt.load(opt.weights_path).items():
        if n in state_dict.keys():
            state_dict[n] = p
        else:
            raise KeyError(n)

    # 将加载的参数应用到模型
    model.load_state_dict(state_dict)
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
    results_data = []

    for i in range(benchmark_len):
        filename = os.path.basename(image_list[i]).split('.')[0]
        descriptions = ''
        print("image:", filename)

        if opt.is_gray:
            input = pil_image.open(image_list[i]).convert('L')
        else:
            input = pil_image.open(image_list[i]).convert('RGB')
        # 保存原图作为Ground Truth
        GT = input
        GT_cal = np.array(input).astype(np.float32) / 255.0
        # 启用高斯噪声
        if opt.gaussian_noise_level is not None:
            if opt.is_gray:
                noise = np.random.normal(0.0, sigma, (input.height, input.width)).astype(np.float32)
                input = np.array(input).astype(np.float32) + noise
                input = np.expand_dims(input, axis=-1)
            else:
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

        # 图像格式转换
        input= jt.array(input).transpose(2, 0, 1).unsqueeze(0)
        # 关闭梯度计算，模型输出去噪结果
        with jt.no_grad():
            pred = model(input)

        # 处理去噪结果
        if opt.is_gray:
            output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).squeeze(0).astype(jt.uint8).numpy()
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

        axes[0].set_title('Ground-Truth', fontsize=8)
        axes[1].set_title('Noisy (PSNR: {:.2f})'.format(noisy_psnr), fontsize=8)
        axes[2].set_title('{} (PSNR: {:.2f})'.format(opt.arch, psnr), fontsize=8)

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
    
    # 添加平均值到结果列表
    results_data.append({
        'filename': 'Average',
        'psnr': round(avg_psnr, 4),
        'ssim': round(avg_ssim, 6)
    })

    # 将结果转换为DataFrame并保存为CSV
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"评估结果已保存至CSV文件: {csv_path}")
