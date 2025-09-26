import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob
import shutil

# def add_gaussian_noise(image, noise_intensity):
#     """添加高斯噪声到图像"""
#     # 将图像转换为浮点数类型
#     image = image.astype(np.float32) / 255.0
#     # 生成高斯噪声
#     noise = np.random.normal(0, noise_intensity/255.0, image.shape)
#     # 添加噪声并裁剪到[0,1]范围
#     noisy_image = np.clip(image + noise, 0, 1)
#     # 转换回uint8格式
#     return (noisy_image * 255).astype(np.uint8)

def add_gaussian_noise(image, noise_intensity):
    """添加高斯噪声到图像（支持单通道和多通道）"""
    # 获取输入图像的通道数
    is_gray = len(image.shape) == 2
    
    # 将图像转换为浮点数类型
    image = image.astype(np.float32) / 255.0
    
    # 生成高斯噪声
    if is_gray:
        noise = np.random.normal(0, noise_intensity/255.0, image.shape)
    else:
        noise = np.random.normal(0, noise_intensity/255.0, image.shape)
    
    # 添加噪声并裁剪到[0,1]范围
    noisy_image = np.clip(image + noise, 0, 1)
    
    # 转换回uint8格式
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    # 如果是单通道噪声，确保输出为二维数组
    if is_gray:
        noisy_image = noisy_image.squeeze()
    
    return noisy_image

def add_gaussian_noise_grayscale(image, noise_intensity):
    """专门为单通道灰度图添加高斯噪声（输入输出均为单通道）"""
    # 确保输入为二维数组（H, W）
    if len(image.shape) == 3:
        image = image.squeeze()
    
    # 添加噪声
    noisy_image = add_gaussian_noise(image, noise_intensity)
    
    # 确保输出为二维数组
    return noisy_image.squeeze()

def process_images_grayscale(dataset_path):
    """处理灰度图像（输入输出均为单通道）"""
    # 定义路径
    base_path = Path(dataset_path)
    original_dir = base_path / "original"
    segmented_dir = base_path / "clean"
    noise_dirs = {
        15: base_path / "noise15",
        25: base_path / "noise25",
        35: base_path / "noise35",
        50: base_path / "noise50"
    }
    
    # 创建输出目录
    segmented_dir.mkdir(parents=True, exist_ok=True)
    for dir_path in noise_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有PNG图像
    image_paths = list(original_dir.glob("*.png"))
    if not image_paths:
        print(f"在 {original_dir} 中没有找到PNG图像")
        return
    
    # 全局计数器
    global_counter = 0
    
    # 处理每张图像
    for img_path in image_paths:
        # 读取图像为灰度图（单通道）
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        # 获取图像尺寸
        h, w = img.shape
        
        # 计算可分割的块数
        h_blocks = h // 256
        w_blocks = w // 256
        
        # 分割图像
        for i in range(h_blocks):
            for j in range(w_blocks):
                # 提取256x256块
                y_start = i * 256
                x_start = j * 256
                patch = img[y_start:y_start+256, x_start:x_start+256]
                
                # 保存干净图像（单通道）
                clean_path = segmented_dir / f"clean_{global_counter}.png"
                cv2.imwrite(str(clean_path), patch)
                
                # 添加不同强度的噪声并保存（单通道）
                for noise_level, noise_dir in noise_dirs.items():
                    noisy_patch = add_gaussian_noise_grayscale(patch.copy(), noise_level)
                    noisy_path = noise_dir / f"noisy_{global_counter}.png"
                    cv2.imwrite(str(noisy_path), noisy_patch)
                
                global_counter += 1
    
    # print(f"灰度图像处理完成! 共生成 {global_counter} 个图像块")
    # print(f"干净灰度图像保存至: {segmented_dir}")
    print(f"grayscale images saved to: {', '.join([str(p) for p in noise_dirs.values()])}")

def process_images(dataset_path):
    # 定义路径
    base_path = Path(dataset_path)
    original_dir = base_path / "original"
    segmented_dir = base_path / "clean"
    noise_dirs = {
        5: base_path / "noise5",
        15: base_path / "noise15",
        25: base_path / "noise25",
        35: base_path / "noise35",
        45: base_path / "noise45"
    }
    
    # 创建输出目录
    segmented_dir.mkdir(parents=True, exist_ok=True)
    for dir_path in noise_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有PNG图像
    image_paths = list(original_dir.glob("*.png"))
    if not image_paths:
        print(f"在 {original_dir} 中没有找到PNG图像")
        return
    
    # 全局计数器
    global_counter = 0
    
    # 处理每张图像
    for img_path in image_paths:
        # 读取图像 (保留原始通道数)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        # 处理单通道图像
        if len(img.shape) == 2:
            # print this if the image is single channel
            print(f"处理单通道图像: {img_path}")
  
            img = cv2.merge([img, img, img])
        elif img.shape[2] == 1:
            img = cv2.merge([img[:,:,0], img[:,:,0], img[:,:,0]])
        elif img.shape[2] == 4:
            # 移除alpha通道
            img = img[:, :, :3]
        
        # 获取图像尺寸
        h, w, _ = img.shape
        
        # 计算可分割的块数
        h_blocks = h // 256
        w_blocks = w // 256
        
        # 分割图像
        for i in range(h_blocks):
            for j in range(w_blocks):
                # 提取256x256块
                y_start = i * 256
                x_start = j * 256
                patch = img[y_start:y_start+256, x_start:x_start+256]
                
                # 保存干净图像
                clean_path = segmented_dir / f"clean_{global_counter}.png"
                cv2.imwrite(str(clean_path), patch)
                
                # 添加不同强度的噪声并保存
                for noise_level, noise_dir in noise_dirs.items():
                    noisy_patch = add_gaussian_noise(patch.copy(), noise_level)
                    noisy_path = noise_dir / f"noisy_{global_counter}.png"
                    cv2.imwrite(str(noisy_path), noisy_patch)
                
                global_counter += 1
    
    print(f"处理完成! 共生成 {global_counter} 个图像块")
    print(f"干净图像保存至: {segmented_dir}")
    print(f"噪声图像保存至: {', '.join([str(p) for p in noise_dirs.values()])}")

def crop_and_save_images_color(src_dir, dst_dir, prefix, patch_size=256):
    """将目录下所有图像无重叠切割为patch_size x patch_size并保存，支持彩色图像"""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    
    img_paths = sorted(glob(os.path.join(src_dir, '*')))
    count = 0
    for img_path in img_paths:
        # 读取彩色图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        h, w = img.shape[:2]  # 高度和宽度
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                if y + patch_size <= h and x + patch_size <= w:
                    patch = img[y:y+patch_size, x:x+patch_size]
                    save_path = os.path.join(dst_dir, f"{prefix}_{count}.png")
                    cv2.imwrite(save_path, patch)
                    count += 1
    return count



def crop_and_save_images(src_dir, dst_dir, prefix, patch_size=256):
    """将目录下所有图像无重叠切割为256x256并保存"""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    
    img_paths = sorted(glob(os.path.join(src_dir, '*')))
    count = 0
    for img_path in img_paths:
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        h, w = img.shape
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                if y + patch_size <= h and x + patch_size <= w:
                    patch = img[y:y+patch_size, x:x+patch_size]
                    save_path = os.path.join(dst_dir, f"{prefix}_{count}.png")
                    cv2.imwrite(save_path, patch)
                    count += 1
    return count

if __name__ == "__main__":
    dataset_path = "dataset_ntire_path"  # 替换为您的实际路径
    process_images(dataset_path)