import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import Unet, pixel_mse_loss
from dataset import Dataset, EvalDataset
from utils import AverageMeter, calc_psnr, calculate_ssim, adjust_lr
from dataset import gen_patch
import copy
import time
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='N2V_Unet')
    parser.add_argument('--images_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/datasets/BSD400')    # 训练集路径
    parser.add_argument('--is_gray', type=str, default='True')  # 是否是灰度图
    parser.add_argument('--clean_valid_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/datasets/BSD68_valid')   # 验证集路径
    parser.add_argument('--outputs_dir', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/weights_50')   # 保存权重路径
    parser.add_argument('--gaussian_noise_level', type=str, default='50') 
    parser.add_argument('--downsampling_factor', type=str, default=None) # 2，3，4
    parser.add_argument('--jpeg_quality', type=int, default=None) # 随意测试
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)  # 从第几轮开始
    parser.add_argument("--resume", default='', type=str)  # 从哪个权重模型继续训练
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay-steps', type=int, default=50)  # 多少轮后开始下降
    parser.add_argument('--lr-decay-gamma', type=float, default=0.5)    # 下降一半
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epoch_save_num', type=int, default=1)  # 每多少轮保存指标
    opt = parser.parse_args()

    if opt.gaussian_noise_level is not None:
        opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))

    if opt.downsampling_factor is not None:
        opt.downsampling_factor = list(map(lambda x: int(x), opt.downsampling_factor.split(',')))

    if opt.jpeg_quality is not None:
        opt.jpeg_quality = list(map(lambda x: int(x), opt.jpeg_quality.split(',')))

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'N2V_Unet':
        if opt.is_gray:
            model = Unet(1,1)
        else:
            model = Unet(3,3)

    model = model.to(device)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))  # 显示信息
            checkpoint = torch.load(opt.resume) # 加载模型
            opt.start_epoch = checkpoint["epoch"] + 1   # 从最后一个检查点获取epoch加1，确定从哪一个epoch开始继续训练
            model.load_state_dict(checkpoint["model"].state_dict()) # 将模型状态字典加载到checkpoint中
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size, opt.gaussian_noise_level, opt.downsampling_factor, opt.jpeg_quality, opt.is_gray)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    eval_dataset = EvalDataset(opt.clean_valid_dir, opt.gaussian_noise_level, opt.downsampling_factor, opt.jpeg_quality, opt.is_gray)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 验证过程变量：最优权重参数、最优epoch、最优psnr
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    results = {'loss': [], 'psnr': [], 'ssim': [], 'time': []}

    for epoch in range(opt.start_epoch, opt.num_epochs):
        epoch_start_time = time.time()
        # Adjust learning rate
        lr = adjust_lr(optimizer, opt.lr, epoch, opt.lr_decay_steps, opt.lr_decay_gamma)

        for param_group in optimizer.param_groups:  # 遍历优化器的参数，更新学习率
            param_group["lr"] = lr

        train_start_time = time.time()

        running_results = {'batch_sizes': 0, 'loss': 0}
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels, pixel_pos = data

                batch_size = inputs.size(0)
                running_results['batch_sizes'] += batch_size

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = pixel_mse_loss(preds, labels, pixel_pos)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        running_results['loss'] = epoch_losses.avg

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        if (epoch + 1) % opt.epoch_save_num == 0:
            state = {"epoch": epoch, "model": model}
            torch.save(state,
                       os.path.join(opt.outputs_dir,
                                    '{}_epoch_{}_{}.pth'.format(opt.arch, epoch, opt.gaussian_noise_level)))

        # ************
        # 验证过程
        # ************
        model.eval()
        epoch_psnr = AverageMeter()  # 记录平均PSNR
        valing_results = {'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}  # 验证集结果字典

        for data in tqdm(eval_dataloader):
            inputs, labels = data
            batch_size = inputs.size(0)
            valing_results['batch_sizes'] += batch_size

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)


            SR = preds.mul(255.0).cpu().numpy().squeeze(0)
            SR = np.clip(SR, 0.0, 255.0).transpose([1, 2, 0])
            SR_y = SR.astype(np.float32)[..., 0] / 255.

            hr_image = labels.mul(255.0).cpu().numpy().squeeze(0)
            hr_image = np.clip(hr_image, 0.0, 255.0).transpose([1, 2, 0])
            hr_y = hr_image.astype(np.float32)[..., 0] / 255.

            epoch_psnr1 = compare_psnr(SR, hr_image, data_range=SR.max() - SR.min())
            epoch_ssim = compare_ssim(SR, hr_image, channel_axis=2, data_range=SR.max() - SR.min())

            valing_results['ssims'] += epoch_ssim * batch_size  # 更新SSIM
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

            epoch_psnr.update(epoch_psnr1)

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        print('eval ssim: {:.4f}'.format(valing_results['ssim']))
        valing_results['psnr'] = epoch_psnr.avg  # psnr是tensor类型，只需要存里面的值

        # 得到最优的psnr
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        epoch_time = round(time.time() - epoch_start_time, 2)

        # 存各种值
        results['loss'].append(running_results['loss'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['time'].append(epoch_time)

        # 每10轮保存一次各种指标，供后续可视化使用
        # TODO 模型继续训练时，注意修改保存的csv文件名，以免覆盖
        if (epoch + 1) % opt.epoch_save_num == 0 and epoch != 0:
            data_frame = pd.DataFrame(
                data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim'], 'Time(s)': results['time']},
                index=range(opt.start_epoch, epoch + 1))
            data_frame.to_csv(opt.outputs_dir + '_srf_' + str(opt.gaussian_noise_level) + '_' + str(
                opt.arch) + '_train_results.csv',
                              index_label='Epoch')
                            
        # 缩进后，每个epoch都保存最优模型，这样随时停止也能得到最优，否则必须等全部训练完才能得到最优模型
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        best_weight_path = 'best_' + str(opt.arch) + '_' + str(opt.gaussian_noise_level) + '.pth'
        torch.save(best_weights, os.path.join(opt.outputs_dir, best_weight_path))