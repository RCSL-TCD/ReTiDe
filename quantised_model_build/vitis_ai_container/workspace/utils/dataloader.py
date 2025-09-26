import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2
from glob import glob
from pathlib import Path
from torchvision.transforms import ToTensor




class DenoiseDatasetGrayScale(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.png')), 
                                key=lambda x: int(Path(x).stem.split('_')[-1]))
        self.clean_paths = sorted(glob(os.path.join(clean_dir, '*.png')), 
                                 key=lambda x: int(Path(x).stem.split('_')[-1]))
        
        assert len(self.noisy_paths) == len(self.clean_paths), "Mismatched clean/noisy image counts"
        for i, (n_path, c_path) in enumerate(zip(self.noisy_paths, self.clean_paths)):
            n_id = Path(n_path).stem.split('_')[-1]
            c_id = Path(c_path).stem.split('_')[-1]
            assert n_id == c_id, f"ID mismatch at index {i}: {n_path} vs {c_path}"
        
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_img = cv2.imread(self.noisy_paths[idx], cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(self.clean_paths[idx], cv2.IMREAD_GRAYSCALE)

        noisy_tensor = self.to_tensor(noisy_img).float()
        clean_tensor = self.to_tensor(clean_img).float()

        return noisy_tensor, clean_tensor
    
class DenoiseDatasetColor(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.png')), 
                                key=lambda x: int(Path(x).stem.split('_')[-1]))
        self.clean_paths = sorted(glob(os.path.join(clean_dir, '*.png')), 
                                 key=lambda x: int(Path(x).stem.split('_')[-1]))
        
        assert len(self.noisy_paths) == len(self.clean_paths), "Mismatched clean/noisy image counts"
        for i, (n_path, c_path) in enumerate(zip(self.noisy_paths, self.clean_paths)):
            n_id = Path(n_path).stem.split('_')[-1]
            c_id = Path(c_path).stem.split('_')[-1]
            assert n_id == c_id, f"ID mismatch at index {i}: {n_path} vs {c_path}"
        
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_img = cv2.imread(self.noisy_paths[idx], cv2.IMREAD_COLOR)
        clean_img = cv2.imread(self.clean_paths[idx], cv2.IMREAD_COLOR)

        noisy_tensor = self.to_tensor(noisy_img).float()
        clean_tensor = self.to_tensor(clean_img).float()

        return noisy_tensor, clean_tensor
    
class TrainingSetGray(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        
        self.clean_paths = sorted(glob(os.path.join(root_dir, "*_clean_*.png")))
        

        self.pairs = []
        for clean_path in self.clean_paths:
            base = os.path.basename(clean_path)
            parts = base.split('_')
            group = parts[0] 
            crop_id = '_'.join(parts[2:])
            noise_path = os.path.join(root_dir, f"{group}_noise_{crop_id}")
            if os.path.exists(noise_path):
                self.pairs.append((clean_path, noise_path))
            else:
                print(f"warning: no images {noise_path}")

        print(f"find {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noise_path = self.pairs[idx]
        noise_img = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noise_img = self.transform(noise_img)
        
        return noise_img, clean_img 
    
def get_grayscale_loaders(data_dir, batch_size=32, val_ratio=0.1, test_ratio=0.1):

    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    full_dataset = TrainingSetGray(data_dir, transform=transform)
    
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader




class MultiNoiseGrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        处理多个噪声级别的灰度图像数据集
        
        目录结构:
        root_dir/
            clean/          # 干净图像
                clean_1.png
                clean_2.png
                ...
            noise15/        # 噪声级别15
                noisy_1.png
                noisy_2.png
                ...
            noise25/        # 噪声级别25
                noisy_1.png
                ...
            ...             # 其他噪声级别文件夹
            
        Args:
            root_dir: 数据集根目录
            transform: 图像转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有噪声级别文件夹
        noise_dirs = glob(os.path.join(root_dir, "noise*"))
        
        # 收集所有噪声-干净图像对
        self.pairs = []
        for noise_dir in noise_dirs:
            # 从文件夹名解析噪声级别 (例如: "noise15" -> 15)
            noise_level = int(os.path.basename(noise_dir).replace("noise", ""))
            
            # 获取该噪声级别下的所有噪声图像
            noise_paths = sorted(glob(os.path.join(noise_dir, "noisy_*.png")))
            
            for noise_path in noise_paths:
                # 从噪声文件名提取编号 (例如: "noisy_123.png" -> "123")
                file_id = os.path.basename(noise_path).split("_")[1].split(".")[0]
                
                # 构建对应的干净图像路径
                clean_path = os.path.join(root_dir, "clean", f"clean_{file_id}.png")
                
                # 检查干净图像是否存在
                if os.path.exists(clean_path):
                    self.pairs.append((noise_path, clean_path, noise_level))
                else:
                    print(f"警告: 找不到对应的干净图像 {clean_path}")
        
        print(f"Found {len(self.pairs)} noise-clean image pairs")
        print(f"including noise level: {set(level for _, _, level in self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noise_path, clean_path, noise_level = self.pairs[idx]
        
        # 以灰度模式读取图像
        noise_img = cv2.imread(noise_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否成功加载
        if noise_img is None:
            raise ValueError(f"无法加载噪声图像: {noise_path}")
        if clean_img is None:
            raise ValueError(f"无法加载干净图像: {clean_path}")
        
        # 应用转换
        if self.transform:
            noise_img = self.transform(noise_img)
            clean_img = self.transform(clean_img)
        
        return noise_img, clean_img


def get_multi_noise_loaders(data_dir, batch_size=32, val_ratio=0.1, test_ratio=0.1):
    """
    获取多噪声级别的数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批大小
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
    ])
    
    # 创建完整数据集
    full_dataset = MultiNoiseGrayDataset(data_dir, transform=transform)
    
    # 计算数据集大小
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Dataloader: train_set={train_size}, val_set={val_size}, test_set={test_size}")
    
    return train_loader, val_loader, test_loader

class MultiNoiseColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        处理多个噪声级别的彩色图像数据集
        
        目录结构:
        root_dir/
            clean/          # 干净图像
                clean_1.png
                clean_2.png
                ...
            noise15/        # 噪声级别15
                noisy_1.png
                noisy_2.png
                ...
            noise25/        # 噪声级别25
                noisy_1.png
                ...
            ...             # 其他噪声级别文件夹
            
        Args:
            root_dir: 数据集根目录
            transform: 图像转换操作
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有噪声级别文件夹
        noise_dirs = glob(os.path.join(root_dir, "noise*"))
        
        # 收集所有噪声-干净图像对
        self.pairs = []
        for noise_dir in noise_dirs:
            # 从文件夹名解析噪声级别 (例如: "noise15" -> 15)
            noise_level = int(os.path.basename(noise_dir).replace("noise", ""))
            
            # 获取该噪声级别下的所有噪声图像
            noise_paths = sorted(glob(os.path.join(noise_dir, "noisy_*.png")))
            
            for noise_path in noise_paths:
                # 从噪声文件名提取编号 (例如: "noisy_123.png" -> "123")
                file_id = os.path.basename(noise_path).split("_")[1].split(".")[0]
                
                # 构建对应的干净图像路径
                clean_path = os.path.join(root_dir, "clean", f"clean_{file_id}.png")
                
                # 检查干净图像是否存在
                if os.path.exists(clean_path):
                    self.pairs.append((noise_path, clean_path, noise_level))
                else:
                    print(f"警告: 找不到对应的干净图像 {clean_path}")
        
        print(f"共找到 {len(self.pairs)} 个噪声-干净图像对")
        print(f"包含的噪声级别: {set(level for _, _, level in self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noise_path, clean_path, noise_level = self.pairs[idx]
        
        # 以彩色模式读取图像 (保留3个通道)
        noise_img = cv2.imread(noise_path, cv2.IMREAD_COLOR)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        
        # 检查图像是否成功加载
        if noise_img is None:
            raise ValueError(f"无法加载噪声图像: {noise_path}")
        if clean_img is None:
            raise ValueError(f"无法加载干净图像: {clean_path}")
        
        # 将BGR转换为RGB (OpenCV默认是BGR, PyTorch需要RGB)
        noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        
        # 应用转换
        if self.transform:
            noise_img = self.transform(noise_img)
            clean_img = self.transform(clean_img)
        
        return noise_img, clean_img


def get_multi_noise_color_loaders(data_dir, batch_size=32, val_ratio=0.1, test_ratio=0.1):
    """
    获取多噪声级别的彩色数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批大小
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor (自动处理HWC->CHW和归一化)
    ])
    
    # 创建完整数据集
    full_dataset = MultiNoiseColorDataset(data_dir, transform=transform)
    
    # 计算数据集大小
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")
    
    return train_loader, val_loader, test_loader