import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def parse_args():
    parser = argparse.ArgumentParser(description='Compare PyTorch and Jittor Metrics')
    parser.add_argument('--pytorch_csv', type=str, default='autodl-tmp/Noise2Void-pytorch-master/Noise2Void-pytorch-master/weights_srf_[25]_N2V_Unet_train_results.csv')
    parser.add_argument('--jittor_csv', type=str, default='autodl-tmp/Noise2Void-jittor-master/Noise2Void-jittor-master/weights_25/results_srf_[25]_N2V_Unet.csv')
    parser.add_argument('--out_dir', type=str, default='autodl-tmp/result')
    parser.add_argument('--model_name', type=str, default='N2V_UNet', help='Name of the model for plot titles')
    return parser.parse_args()

def load_metrics_data(csv_path):
    """Load metrics data from CSV file"""
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    required_columns = ['Epoch', 'Loss', 'PSNR', 'SSIM', 'Time(s)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件 {csv_path} 缺少必要的列: {missing_cols}")
    
    # 将所有数值列转换为数值类型
    numeric_cols = ['Epoch', 'Loss', 'PSNR', 'SSIM', 'Time(s)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN值的行
    df = df.dropna(subset=numeric_cols)
    
    # 确保只使用前300个epoch的数据
    df = df[df['Epoch'] <= 300]
    
    # 计算累计时间（如果Time(s)是每个epoch的耗时）
    if 'Cumulative_Time(s)' not in df.columns:
        df['Cumulative_Time(s)'] = df['Time(s)'].cumsum()
    
    # 添加调试信息
    print(f"加载数据: {csv_path}")
    print(f"数据类型:\n{df.dtypes}")
    print(f"数据行数: {len(df)}")
    
    return df

def plot_metric_comparison(pytorch_data, jittor_data, metric, model_name, save_dir):
    """绘制单个指标的对比曲线"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 指标配置
    metric_config = {
        'Loss': {'ylabel': 'Loss', 'title': 'Loss Comparison', 'ylim': None},
        'PSNR': {'ylabel': 'PSNR (dB)', 'title': 'PSNR Comparison', 'ylim': None},
        'SSIM': {'ylabel': 'SSIM', 'title': 'SSIM Comparison', 'ylim': [0, 1]},
        'Cumulative_Time(s)': {'ylabel': 'Cumulative Time (seconds)', 'title': 'Training Time Comparison', 'ylim': None}
    }
    
    # 绘制曲线
    ax.plot(pytorch_data['Epoch'], pytorch_data[metric], 
            label='PyTorch', color='blue', linewidth=2, marker='o', markersize=4)
    ax.plot(jittor_data['Epoch'], jittor_data[metric], 
            label='Jittor', color='red', linewidth=2, marker='s', markersize=4)
    
    # 设置坐标轴
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_config[metric]['ylabel'], fontsize=12)
    ax.set_title(f'{metric_config[metric]["title"]}: {model_name} (300 Epochs)', fontsize=14, pad=20)
    
    # 设置Y轴范围（针对SSIM等有固定范围的指标）
    if metric_config[metric]['ylim']:
        ax.set_ylim(metric_config[metric]['ylim'])
    
    # 设置X轴刻度间隔为50
    x_major_locator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_major_locator)
    
    # 添加图例和网格
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{model_name}_{metric.lower()}_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"图表已保存至 {save_path}")
    
    plt.close()

def print_final_metrics(pytorch_data, jittor_data):
    """打印最后一个epoch的指标对比"""
    # 获取最后一个epoch的数据
    pt_last = pytorch_data.iloc[-1]
    jt_last = jittor_data.iloc[-1]
    
    # 确保两个框架的epoch数相同
    if pt_last['Epoch'] != jt_last['Epoch']:
        print(f"警告: PyTorch最后epoch为{pt_last['Epoch']}, Jittor最后epoch为{jt_last['Epoch']}")
        # 使用两者中较小的epoch作为比较点
        common_epoch = min(pt_last['Epoch'], jt_last['Epoch'])
        pt_last = pytorch_data[pytorch_data['Epoch'] == common_epoch].iloc[-1]
        jt_last = jittor_data[jittor_data['Epoch'] == common_epoch].iloc[-1]
    
    print("\n最终指标 (在Epoch {:.0f}时):".format(pt_last['Epoch']))
    print(f"{'指标':<15} | {'PyTorch':<15} | {'Jittor':<15} | {'差异':<15}")
    print("-" * 65)
    
    # 计算并打印各个指标
    metrics = [
        ('Loss', lambda x: f"{x:.6f}"),
        ('PSNR', lambda x: f"{x:.2f} dB"),
        ('SSIM', lambda x: f"{x:.4f}"),
        ('Cumulative_Time(s)', lambda x: f"{x:.2f} s")
    ]
    
    for metric, formatter in metrics:
        try:
            pt_val = pt_last[metric]
            jt_val = jt_last[metric]
            diff = pt_val - jt_val
            
            # 格式化差异
            if metric == 'Loss' or metric == 'Cumulative_Time(s)':
                diff_str = f"{diff:.2f}"
            else:  # PSNR和SSIM越高越好
                diff_str = f"{diff:.2f}"
            
            print(f"{metric:<15} | {formatter(pt_val):<15} | {formatter(jt_val):<15} | {diff_str:<15}")
        except Exception as e:
            print(f"处理指标 {metric} 时出错: {e}")
            print(f"PyTorch值: {pt_last[metric]} (类型: {type(pt_last[metric])})")
            print(f"Jittor值: {jt_last[metric]} (类型: {type(jt_last[metric])})")

def main():
    args = parse_args()
    
    try:
        # 加载数据
        print("加载PyTorch指标...")
        pytorch_data = load_metrics_data(args.pytorch_csv)
        
        print("加载Jittor指标...")
        jittor_data = load_metrics_data(args.jittor_csv)
        
        # 确保两个数据集的epoch范围一致
        min_epochs = min(pytorch_data['Epoch'].max(), jittor_data['Epoch'].max())
        if min_epochs < 300:
            print(f"警告: 只有 {min_epochs} epochs 可用于比较 (需要300个)")
        
        # 打印最终指标对比
        print_final_metrics(pytorch_data, jittor_data)
        
        # 绘制四个指标的对比曲线
        metrics = ['Loss', 'PSNR', 'SSIM', 'Cumulative_Time(s)']
        for metric in metrics:
            plot_metric_comparison(pytorch_data, jittor_data, metric, args.model_name, args.out_dir)
        
        print("所有操作已完成!")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()