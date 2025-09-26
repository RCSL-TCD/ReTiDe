import torch
import numpy as np
from tqdm import tqdm
import cv2
import os

def calculate_psnr_tensor(img1, img2, data_range=1.0):
    """计算两个张量之间的PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse)).item()

def calculate_actual_noise_level(noisy_img, clean_img):
    """
    计算实际噪声水平（标准差）
    使用噪声图像与干净图像之间的残差
    """
    # 计算残差
    residual = noisy_img - clean_img
    
    # 计算残差的标准差
    return torch.std(residual).item()


def estimate_noise_level(image):
    """
    估计图像的噪声水平（标准差）
    支持批量图像和单个图像
    自动处理CUDA张量
    """
    # 如果是批量图像 [B, C, H, W]
    if image.dim() == 4:
        noise_levels = []
        for i in range(image.size(0)):
            # 递归处理每个图像
            noise_levels.append(estimate_noise_level(image[i]))
        return sum(noise_levels) / len(noise_levels)
    
    # 单张图像处理 [C, H, W]
    if image.dim() == 3:
        # 确保张量在CPU上
        if image.is_cuda:
            image = image.cpu()
        
        # 转换为HWC格式的NumPy数组
        image = image.permute(1, 2, 0).numpy()
        
        # 确保图像数据类型是uint8
        if image.dtype != np.uint8:
            # 缩放并转换为uint8
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # 转换为灰度图像
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = image
    
    # 使用拉普拉斯算子估计噪声水平
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return np.std(laplacian)

def test_denoising_model(model, dataloader, device, output_noise=False, output_path=None):
    """
    测试降噪模型性能
    
    参数:
    model: 训练好的降噪模型
    dataloader: 测试数据加载器
    device: 计算设备 (cuda/cpu)
    output_noise: 模型是否输出噪声图 (True) 或干净图像 (False)
    
    返回:
    results: 包含测试结果的字典
    """
    model.eval()
    model.to(device)
    
    # 初始化统计量
    total_psnr_noisy = 0.0
    total_psnr_denoised = 0.0
    num_samples = 0
    
    # 噪声水平统计 (归一化范围)
    sum_input_noise_norm = 0.0
    sum_output_noise_norm = 0.0
    
    # 总体噪声水平统计 (更准确的方法)
    sum_squared_input_noise = 0.0
    sum_squared_output_noise = 0.0
    total_pixels = 0
    
    # 用于计算PSNR的MSE统计
    sum_mse_noisy = 0.0
    sum_mse_denoised = 0.0
    
    with torch.no_grad():
        # 添加全局样本计数器
        global_idx = 0
        
        for noisy_imgs, clean_imgs, _ in tqdm(dataloader, desc="Testing"):
            # 移动到设备
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 模型推理
            outputs = model(noisy_imgs)
            
            # 根据模型输出类型处理结果
            if output_noise:
                # 模型输出噪声图，需要计算去噪图像
                denoised_imgs = torch.clamp(noisy_imgs - outputs, 0, 1)
            else:
                # 模型直接输出去噪图像
                denoised_imgs = torch.clamp(outputs, 0, 1)
            
            # 计算残差
            input_residual = noisy_imgs - clean_imgs
            output_residual = denoised_imgs - clean_imgs
            
            # 计算噪声水平 (归一化范围)
            input_noise_level_norm = torch.std(input_residual).item()
            output_noise_level_norm = torch.std(output_residual).item()
            
            # 累加噪声水平
            sum_input_noise_norm += input_noise_level_norm * noisy_imgs.size(0)
            sum_output_noise_norm += output_noise_level_norm * noisy_imgs.size(0)
            
            # 为总体噪声水平计算收集数据
            sum_squared_input_noise += torch.sum(input_residual ** 2).item()
            sum_squared_output_noise += torch.sum(output_residual ** 2).item()
            total_pixels += input_residual.numel()
            
            # 计算PSNR
            batch_size = noisy_imgs.size(0)
            for i in range(batch_size):
                # 带噪图像PSNR
                psnr_noisy = calculate_psnr_tensor(noisy_imgs[i], clean_imgs[i])
                
                # 去噪图像PSNR
                psnr_denoised = calculate_psnr_tensor(denoised_imgs[i], clean_imgs[i])
                
                # 累加统计量
                total_psnr_noisy += psnr_noisy
                total_psnr_denoised += psnr_denoised
                num_samples += 1
            
            # 计算MSE用于总体PSNR
            sum_mse_noisy += torch.sum((noisy_imgs - clean_imgs) ** 2).item()
            sum_mse_denoised += torch.sum((denoised_imgs - clean_imgs) ** 2).item()


            # 输出图像保存
            if output_path:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
            # 创建输出目录
            if not os.path.exists(output_path + "/noisy"):
                os.makedirs(output_path + "/noisy")
            if not os.path.exists(output_path + "/clean"):
                os.makedirs(output_path + "/clean")
            if not os.path.exists(output_path + "/denoised"):
                os.makedirs(output_path + "/denoised")

            # 确保输出张量在CPU上
            noisy_imgs_cpu = noisy_imgs.cpu()
            denoised_imgs_cpu = denoised_imgs.cpu()
            clean_imgs_cpu = clean_imgs.cpu()
            
            # 转换为NumPy数组并归一化到[0, 255]
            # 注意：这里使用新的变量名避免覆盖
            noisy_imgs_np = noisy_imgs_cpu.numpy().transpose(0, 2, 3, 1) * 255.0
            denoised_imgs_np = denoised_imgs_cpu.numpy().transpose(0, 2, 3, 1) * 255.0
            clean_imgs_np = clean_imgs_cpu.numpy().transpose(0, 2, 3, 1) * 255.0
            
            # 确保数据类型是uint8
            noisy_imgs_np = noisy_imgs_np.astype(np.uint8)
            denoised_imgs_np = denoised_imgs_np.astype(np.uint8)
            clean_imgs_np = clean_imgs_np.astype(np.uint8)

            # 保存噪声图像（修复部分：直接使用NumPy数组）
            for i in range(batch_size):
                # 使用全局唯一的索引
                idx = global_idx + i
                
                # 直接使用转换后的NumPy数组
                noisy_img = noisy_imgs_np[i]
                denoised_img = denoised_imgs_np[i]
                clean_imgs_img = clean_imgs_np[i]

                # 根据通道数处理图像保存
                if noisy_img.shape[-1] == 1:  # 单通道图像 (灰度)
                    noisy_img = noisy_img.squeeze(axis=-1)  # 移除通道维度 [H, W, 1] -> [H, W]
                else:  # 多通道图像 (通常是RGB)
                    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
                
                if denoised_img.shape[-1] == 1:
                    denoised_img = denoised_img.squeeze(axis=-1)
                else:
                    denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)

                if clean_imgs_img.shape[-1] == 1:
                    clean_imgs_img = clean_imgs_img.squeeze(axis=-1)
                else:
                    clean_imgs_img = cv2.cvtColor(clean_imgs_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f"{output_path}/noisy/noisy_{idx}.png", noisy_img)
                #cv2.imwrite(f"{output_path}/clean/clean_{idx}.png", clean_imgs[i].cpu().numpy().transpose(1, 2, 0) * 255.0)
                cv2.imwrite(f"{output_path}/denoised/denoised_{idx}.png", denoised_img)
                cv2.imwrite(f"{output_path}/clean/clean_{idx}.png", clean_imgs_img)

            # 更新全局索引
            global_idx += batch_size


    # 计算平均值
    avg_psnr_noisy = total_psnr_noisy / num_samples
    avg_psnr_denoised = total_psnr_denoised / num_samples
    
    # 计算平均噪声水平 (归一化范围)
    avg_input_noise_norm = sum_input_noise_norm / num_samples
    avg_output_noise_norm = sum_output_noise_norm / num_samples
    
    # 计算总体噪声水平 (更准确的方法)
    overall_input_noise_norm = (sum_squared_input_noise / total_pixels) ** 0.5
    overall_output_noise_norm = (sum_squared_output_noise / total_pixels) ** 0.5
    
    # 计算总体PSNR (更准确的方法)
    mse_noisy = sum_mse_noisy / total_pixels
    mse_denoised = sum_mse_denoised / total_pixels
    overall_psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy)) if mse_noisy > 0 else float('inf')
    overall_psnr_denoised = 20 * np.log10(1.0 / np.sqrt(mse_denoised)) if mse_denoised > 0 else float('inf')
    
    # 计算PSNR提升
    psnr_gain = avg_psnr_denoised - avg_psnr_noisy
    overall_psnr_gain = overall_psnr_denoised - overall_psnr_noisy
    
    # 转换为[0,255]范围
    avg_input_noise_255 = avg_input_noise_norm * 255
    avg_output_noise_255 = avg_output_noise_norm * 255
    overall_input_noise_255 = overall_input_noise_norm * 255
    overall_output_noise_255 = overall_output_noise_norm * 255
    
    # save the results to file
    if output_path:
        results_file = os.path.join(output_path, "results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Average PSNR (Noisy): {avg_psnr_noisy:.2f} dB\n")
            f.write(f"Average PSNR (Denoised): {avg_psnr_denoised:.2f} dB\n")
            f.write(f"PSNR Gain: {psnr_gain:.2f} dB\n")
            f.write(f"Overall PSNR (Noisy): {overall_psnr_noisy:.2f} dB\n")
            f.write(f"Overall PSNR (Denoised): {overall_psnr_denoised:.2f} dB\n")
            f.write(f"Overall PSNR Gain: {overall_psnr_gain:.2f} dB\n")
            f.write(f"Average Input Noise Level (255): {avg_input_noise_255:.2f}\n")

    # 返回结果
    results = {
        # 样本信息
        "num_samples": num_samples,
        "total_pixels": total_pixels,
        
        # 噪声水平 - 归一化范围 [0,1]
        "avg_input_noise_norm": avg_input_noise_norm,
        "avg_output_noise_norm": avg_output_noise_norm,
        "overall_input_noise_norm": overall_input_noise_norm,
        "overall_output_noise_norm": overall_output_noise_norm,
        
        # 噪声水平 - 原始范围 [0,255]
        "avg_input_noise_255": avg_input_noise_255,
        "avg_output_noise_255": avg_output_noise_255,
        "overall_input_noise_255": overall_input_noise_255,
        "overall_output_noise_255": overall_output_noise_255,
        
        # PSNR指标
        "avg_psnr_noisy": avg_psnr_noisy,
        "avg_psnr_denoised": avg_psnr_denoised,
        "overall_psnr_noisy": overall_psnr_noisy,
        "overall_psnr_denoised": overall_psnr_denoised,
        
        # PSNR提升
        "psnr_gain": psnr_gain,
        "overall_psnr_gain": overall_psnr_gain,
        
        # 残差统计
        "sum_squared_input_noise": sum_squared_input_noise,
        "sum_squared_output_noise": sum_squared_output_noise,
        "mse_noisy": mse_noisy,
        "mse_denoised": mse_denoised
    }
    
    return results

def print_test_results(results):
    """打印测试结果 - 包含两种范围的噪声水平"""
    print("\n" + "="*70)
    print("Denoising Model Test Results - Comprehensive Report")
    print("="*70)
    print(f"Tested on {results['num_samples']} samples ({results['total_pixels']/1e6:.2f}M pixels)")
    
    # 噪声水平报告
    print("\n[Noise Levels]")
    # print("  Normalized Range [0,1]:")
    # print(f"    Avg Input Noise: {results['avg_input_noise_norm']:.4f}")
    # print(f"    Avg Output Noise: {results['avg_output_noise_norm']:.4f}")
    # print(f"    Overall Input Noise: {results['overall_input_noise_norm']:.4f} (most accurate)")
    # print(f"    Overall Output Noise: {results['overall_output_noise_norm']:.4f} (most accurate)")
    
    print("Original Range [0,255]:")
    print(f"    Avg Input Noise: {results['avg_input_noise_255']:.2f}")
    print(f"    Avg Output Noise: {results['avg_output_noise_255']:.2f}")
    print(f"    Overall Input Noise: {results['overall_input_noise_255']:.2f} (most accurate)")
    print(f"    Overall Output Noise: {results['overall_output_noise_255']:.2f} (most accurate)")
    
    # PSNR报告
    print("\n[PSNR Metrics]")
    print("  Per-Image Averages:")
    print(f"    PSNR (Noisy): {results['avg_psnr_noisy']:.2f} dB")
    print(f"    PSNR (Denoised): {results['avg_psnr_denoised']:.2f} dB")
    print(f"    PSNR Gain: {results['psnr_gain']:.2f} dB")
    
    # print("\n  Overall (Per-Pixel):")
    # print(f"    PSNR (Noisy): {results['overall_psnr_noisy']:.2f} dB")
    # print(f"    PSNR (Denoised): {results['overall_psnr_denoised']:.2f} dB")
    # print(f"    PSNR Gain: {results['overall_psnr_gain']:.2f} dB")
    
    # 噪声降低百分比
    # noise_reduction_pct = (1 - results['overall_output_noise_norm'] / results['overall_input_noise_norm']) * 100
    # print(f"\nNoise Reduction: {noise_reduction_pct:.2f}%")
    
    print("="*70)
    
    # 返回最重要的指标
    return {
        "overall_input_noise_255": results["overall_input_noise_255"],
        "overall_output_noise_255": results["overall_output_noise_255"],
        "overall_psnr_denoised": results["overall_psnr_denoised"],
        "overall_psnr_gain": results["overall_psnr_gain"]#,
        #"noise_reduction_pct": noise_reduction_pct
    }


# def test_denoising_model_grayscale(model, dataloader, device, output_path):
#     """执行模型推理并保存结果，支持灰度和彩色图像"""
#     os.makedirs(output_path, exist_ok=True)
#     model.eval()
#     model.to(device)
    
#     global_idx = 0  # 全局计数器，确保从0开始连续命名
#     with torch.no_grad():
#         for noisy_imgs, _ in tqdm(dataloader, desc="Denoising"):
#             noisy_imgs = noisy_imgs.to(device)
#             denoised = model(noisy_imgs)
            
#             # 保存输出图像
#             for j in range(denoised.shape[0]):
#                 img = denoised[j].cpu().clamp(0, 1).numpy()
                
#                 if img.shape[0] == 1:
#                     # 灰度图像：[1, H, W] -> [H, W]
#                     img = (img[0] * 255).astype(np.uint8)
#                 elif img.shape[0] == 3:
#                     # 彩色图像：[3, H, W] -> [H, W, 3]
#                     img = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
#                 else:
#                     raise ValueError(f"不支持的通道数：{img.shape[0]}，只能处理灰度图或RGB图像")
                
#                 save_path = os.path.join(output_path, f"denoised_{global_idx}.png")
#                 cv2.imwrite(save_path, img)
#                 global_idx += 1
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

def calc_psnr(img1, img2, max_val=1.0):
    """计算两张图像的PSNR值"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val**2 / mse)

def test_denoising_model_grayscale(model, dataloader, device, output_path):
    """执行模型推理并保存结果，支持灰度和彩色图像，同时输出PSNR"""
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    model.to(device)
    
    global_idx = 0  # 全局计数器
    psnr_results = []  # 存储PSNR结果
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs in tqdm(dataloader, desc="Denoising"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 模型推理
            denoised = model(noisy_imgs).clamp(0, 1)
            
            # 保存输出图像并计算PSNR
            for j in range(denoised.shape[0]):
                # 处理降噪后图像
                denoised_np = denoised[j].cpu().numpy()
                clean_np = clean_imgs[j].cpu().numpy()
                noisy_np = noisy_imgs[j].cpu().numpy()
                
                # 根据通道数调整维度
                if denoised_np.shape[0] == 1:  # 灰度图像
                    denoised_np = denoised_np[0]
                    clean_np = clean_np[0]
                    noisy_np = noisy_np[0]
                elif denoised_np.shape[0] == 3:  # 彩色图像
                    denoised_np = np.transpose(denoised_np, (1, 2, 0))
                    clean_np = np.transpose(clean_np, (1, 2, 0))
                    noisy_np = np.transpose(noisy_np, (1, 2, 0))
                else:
                    raise ValueError(f"不支持的通道数：{denoised_np.shape[0]}")
                
                # 计算PSNR
                psnr_before = calc_psnr(noisy_np, clean_np)
                psnr_after = calc_psnr(denoised_np, clean_np)
                psnr_results.append((psnr_before, psnr_after))
                
                # 保存降噪后图像
                save_path = os.path.join(output_path, f"denoised_{global_idx}.png")
                cv2.imwrite(save_path, (denoised_np * 255).astype(np.uint8))
                
                global_idx += 1
    
    # 计算并打印平均PSNR
    psnr_before_avg = np.mean([x[0] for x in psnr_results])
    psnr_after_avg = np.mean([x[1] for x in psnr_results])
    print(f"\nPSNR - before: {psnr_before_avg:.2f} dB | after: {psnr_after_avg:.2f} dB")
    print(f"PSNR improvement: {psnr_after_avg - psnr_before_avg:.2f} dB")
    
    return psnr_results

# 使用示例
if __name__ == "__main__":
    # 假设已经定义好模型和数据加载器
    from models import DenoisingUNet  # 你的UNet模型
    from dataloader import create_dataloader
    
    # 配置参数
    DATA_ROOT = "dataset_ntire_path"
    NOISE_LEVEL = 25
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型和数据加载器
    model = DenoisingUNet(in_channels=3, out_channels=3)
    dataloader = create_dataloader(
        root_dir=DATA_ROOT,
        noise_level=NOISE_LEVEL,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # 运行测试
    results = test_denoising_model(
        model=model,
        dataloader=dataloader,
        device=DEVICE,
        output_noise=False  # 根据模型实际输出设置
    )
    
    # 打印结果
    print_test_results(results)