import argparse
import os
import time  # 导入时间模块
import jittor as jt
from jittor import nn
from jittor.optim import Adam
from jittor.dataset import DataLoader
from tqdm import tqdm
from model import Unet, pixel_mse_loss
from dataset import Dataset, EvalDataset
import copy
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

jt.flags.use_cuda = 1

class AverageMeter:
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

def adjust_lr(optimizer, base_lr, epoch, lr_decay_steps, lr_decay_gamma):
    lr = base_lr * (lr_decay_gamma ** (epoch // lr_decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='N2V_Unet')
    parser.add_argument('--images_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/datasets/BSD400')
    parser.add_argument('--is_gray', type=str, default='True')
    parser.add_argument('--clean_valid_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/datasets/BSD68_valid')
    parser.add_argument('--outputs_dir', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/weights_50')
    parser.add_argument('--gaussian_noise_level', type=str, default='50')
    parser.add_argument('--downsampling_factor', type=str, default=None)
    parser.add_argument('--jpeg_quality', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument("--resume", default='', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay_steps', type=int, default=50)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epoch_save_num', type=int, default=1)
    opt = parser.parse_args()

    if opt.gaussian_noise_level is not None:
        opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))
    
    if opt.downsampling_factor is not None:
        opt.downsampling_factor = list(map(lambda x: int(x), opt.downsampling_factor.split(',')))
    
    if opt.jpeg_quality is not None:
        opt.jpeg_quality = list(map(lambda x: int(x), opt.jpeg_quality.split(',')))
    
    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)
    
    jt.set_global_seed(opt.seed)
    
    
    if opt.arch == 'N2V_Unet':
        model = Unet(1,1) if opt.is_gray == 'True' else Unet(3,3)
    
    if opt.resume and os.path.isfile(opt.resume):
        print(f"=> loading checkpoint '{opt.resume}'")
        checkpoint = jt.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
    
    optimizer = Adam(model.parameters(), lr=opt.lr)
    
    dataset = Dataset(opt.images_dir, opt.patch_size, opt.gaussian_noise_level, 
                     opt.downsampling_factor, opt.jpeg_quality, opt.is_gray=='True')
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, 
                           num_workers=opt.threads, drop_last=True)
    
    eval_dataset = EvalDataset(opt.clean_valid_dir, opt.gaussian_noise_level, 
                              opt.downsampling_factor, opt.jpeg_quality, opt.is_gray=='True')
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    # 在结果字典中添加运行时间记录
    results = {'loss': [], 'psnr': [], 'ssim': [], 'time': []}
    
    for epoch in range(opt.start_epoch, opt.num_epochs):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        lr = adjust_lr(optimizer, opt.lr, epoch, opt.lr_decay_steps, opt.lr_decay_gamma)
        
        model.train()
        epoch_losses = AverageMeter()
        
        with tqdm(total=len(dataloader), desc=f'epoch {epoch+1}/{opt.num_epochs}') as pbar:
            for inputs, labels, pixel_pos in dataloader:
                inputs = jt.array(inputs)
                labels = jt.array(labels)
                pixel_pos = jt.array(pixel_pos)
                
                preds = model(inputs)
                loss = pixel_mse_loss(preds, labels, pixel_pos)
                
                optimizer.step(loss)
                epoch_losses.update(loss.item(), inputs.shape[0])
                
                pbar.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                pbar.update(1)
        
        if (epoch + 1) % opt.epoch_save_num == 0:
            state = {"epoch": epoch, "model": model.state_dict()}
            jt.save(state, os.path.join(opt.outputs_dir, 
                      f'{opt.arch}_epoch_{epoch}_{opt.gaussian_noise_level}.pth'))
        
        model.eval()
        epoch_psnr = AverageMeter()
        ssim_total = 0.0
        count = 0
        
        for inputs, labels in tqdm(eval_dataloader, desc='Evaluating'):
            inputs = jt.array(inputs)
            labels = jt.array(labels)
            
            with jt.no_grad():
                preds = model(inputs)
            
            sr = preds.numpy() * 255.0
            hr = labels.numpy() * 255.0
            
            sr = np.clip(sr, 0, 255).squeeze(0).transpose(1, 2, 0)
            hr = np.clip(hr, 0, 255).squeeze(0).transpose(1, 2, 0)
            
            psnr_val = compare_psnr(hr, sr, data_range=255)
            ssim_val = compare_ssim(hr, sr, channel_axis=2, data_range=255)
            
            epoch_psnr.update(psnr_val)
            ssim_total += ssim_val
            count += 1
        
        avg_ssim = ssim_total / count
        print(f'eval psnr: {epoch_psnr.avg:.2f}, ssim: {avg_ssim:.4f}')
        
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
        
        # 计算并记录当前epoch的运行时间（秒）
        epoch_time = time.time() - epoch_start_time
        results['loss'].append(epoch_losses.avg)
        results['psnr'].append(epoch_psnr.avg)
        results['ssim'].append(avg_ssim)
        results['time'].append(epoch_time)  # 保存当前epoch的运行时间
        
        if (epoch + 1) % opt.epoch_save_num == 0 and epoch != 0:
            df = pd.DataFrame({
                'Loss': results['loss'],
                'PSNR': results['psnr'],
                'SSIM': results['ssim'],
                'Time (s)': results['time']  # 添加时间列到DataFrame
            }, index=range(opt.start_epoch, epoch+1))
            csv_name = f'{opt.outputs_dir}/results_srf_{opt.gaussian_noise_level}_{opt.arch}.csv'
            df.to_csv(csv_name, index_label='Epoch')
        
        print(f'best epoch: {best_epoch}, psnr: {best_psnr:.2f}, current epoch time: {epoch_time:.2f}s')
        best_path = f'best_{opt.arch}_{opt.gaussian_noise_level}.pth'
        jt.save(best_weights, os.path.join(opt.outputs_dir, best_path))
