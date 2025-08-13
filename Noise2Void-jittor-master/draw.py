import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import jittor as jt 

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description='Draw Evaluation')
parser.add_argument('--arch', type=str, default='N2V_Unet')
parser.add_argument('--csv_dir', default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/weights', type=str)
parser.add_argument('--out_dir', default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/Plt/', type=str)
parser.add_argument('--gaussian_noise_level', type=str, default='25')
opt = parser.parse_args()

if opt.gaussian_noise_level is not None:
    opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))

csv_file_path = opt.csv_dir +'_srf_' + str(opt.gaussian_noise_level) + '_' + str(opt.arch) + '_train_result.csv'
data_frame = pd.read_csv(csv_file_path, index_col='Epoch')

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# 绘制 Loss 曲线
axs[0].plot(data_frame.index, data_frame['Loss'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='blue')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend(loc='upper right')
axs[0].grid(ls='--')

# 绘制 PSNR 曲线
axs[1].plot(data_frame.index, data_frame['PSNR'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='green')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('PSNR(dB)')
axs[1].legend(loc='lower right')
axs[1].grid(ls='--')

# 绘制 SSIM 曲线
axs[2].plot(data_frame.index, data_frame['SSIM'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='red')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('SSIM')
axs[2].legend(loc='lower right')
axs[2].grid(ls='--')

# 调整子图间距并保存原有三指标图
plt.tight_layout()
plt.autoscale(enable=True, axis='y', tight=True)
plt_save_path = opt.out_dir + 'evalution_plt_{}_{}'.format(opt.gaussian_noise_level, opt.arch)
os.makedirs(opt.out_dir, exist_ok=True)
plt.savefig(plt_save_path, bbox_inches='tight', dpi=600)
plt.close()

# 绘制 Time(s) 曲线
plt.figure(figsize=(6, 4)) 
plt.plot(data_frame.index, data_frame['Time(s)'], 
         label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), 
         color='purple')
plt.xlabel('Epoch')
plt.ylabel('Time(s)')
plt.legend(loc='upper right')
plt.grid(ls='--')

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10))

time_save_path = opt.out_dir + 'time_plt_{}_{}'.format(opt.gaussian_noise_level, opt.arch)
plt.tight_layout()
plt.savefig(time_save_path, bbox_inches='tight', dpi=600)

plt.show()
