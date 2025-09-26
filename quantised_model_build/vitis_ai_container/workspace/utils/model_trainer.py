import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import math

import torch
import math
from tqdm import tqdm

def calculate_raw_metrics(test_loader, device='cuda'):
    """
    计算测试集上噪声图像与干净图像之间的原始MSE和PSNR
    (不经过任何模型处理)
    
    参数:
        test_loader: 测试数据加载器
        device: 计算设备
        
    返回:
        mse: 平均均方误差
        psnr: 平均峰值信噪比 (dB)
    """
    total_mse = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for noise_imgs, clean_imgs in tqdm(test_loader, desc="original metrics"):
            # 将数据移动到指定设备
            noise_imgs = noise_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # 计算当前batch的MSE
            batch_mse = torch.mean((noise_imgs - clean_imgs) ** 2)
            total_mse += batch_mse.item() * noise_imgs.size(0)  # 加权求和
            
            # 计算当前batch的PSNR
            if batch_mse.item() > 0:
                batch_psnr = 10 * math.log10(1.0 / batch_mse.item())
            else:
                batch_psnr = float('inf')  # 如果MSE为0，PSNR无限大
            total_psnr += batch_psnr * noise_imgs.size(0)
            
            num_samples += noise_imgs.size(0)
    
    # 计算平均值
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    
    return avg_mse, avg_psnr


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda', lr=0.00000001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.best_val_loss = float('inf')
        # self.last_model = None
        os.makedirs('checkpoints', exist_ok=True)

    def train_epoch(self, epoch, model=None):
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model.to(self.device)
        self.model.train()
        train_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}', leave=False)
        for batch_idx, (noise_imgs, clean_imgs) in enumerate(progress_bar):
            noise_imgs = noise_imgs.to(self.device)
            clean_imgs = clean_imgs.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(noise_imgs)
            loss = self.criterion(outputs, clean_imgs)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(self.train_loader)
        elapsed_time = time.time() - start_time
        
        return train_loss, elapsed_time

    def validate(self, epoch, model=None):
        if model is not None:
            model.eval()
        else:
            model = self.model.eval()
        val_loss = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Validate Epoch {epoch}', leave=False)
            for noise_imgs, clean_imgs in progress_bar:
                noise_imgs = noise_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                outputs = model(noise_imgs)
                loss = self.criterion(outputs, clean_imgs)
                val_loss += loss.item()
                progress_bar.set_postfix({'val_loss': loss.item()})
        
        val_loss /= len(self.val_loader)
        elapsed_time = time.time() - start_time
        
        self.scheduler.step(val_loss)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), f'checkpoints/best_model_epoch{epoch}.pth')
            print(f"\nNew best model saved with val_loss: {val_loss:.4f}")
        
        return val_loss, elapsed_time

    def test(self, model = None):

        if model is not None:
            model.eval()
        else:
            model = self.model.eval()
        test_loss = 0.0
        start_time = time.time()
        
        # best_model_path = sorted([f for f in os.listdir('checkpoints') if f.startswith('best_model')])[-1]
        # self.model.load_state_dict(torch.load(os.path.join('checkpoints', best_model_path)))
        # print(f"\nLoaded best model: {best_model_path}")
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Testing', leave=False)
            for noise_imgs, clean_imgs in progress_bar:
                noise_imgs = noise_imgs.to(self.device)
                clean_imgs = clean_imgs.to(self.device)
                
                outputs = model(noise_imgs)
                loss = self.criterion(outputs, clean_imgs)
                test_loss += loss.item()
                progress_bar.set_postfix({'test_loss': loss.item()})
        
        test_loss /= len(self.test_loader)
        mse = test_loss  # Mean Squared Error is the loss itself
        psnr = 10 * math.log10(1.0 / test_loss)
        # print(f"\nTest Loss: {test_loss:.4f} | PSNR: {psnr:.2f} dB")
        # print(f"Test completed in {time.time() - start_time:.2f} seconds")
        elapsed_time = time.time() - start_time
        
        return mse, psnr, elapsed_time

    def train(self, epochs, model=None):
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model.to(self.device)
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}\n")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_time = self.train_epoch(epoch, self.model)
            val_loss, val_time = self.validate(epoch, self.model)
            
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"Train Loss: {train_loss:.8f} | Time: {train_time:.2f}s")
            print(f"Val Loss: {val_loss:.8f} | Time: {val_time:.2f}s")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f'checkpoints/model_epoch{epoch}.pth')
                print(f"Model saved at epoch {epoch}")
        
        print("\nTraining completed. Running test...")
        mse, psnr, elapsed_time = self.test()
        print(f"Test MSE: {mse:.4f} | PSNR: {psnr:.4f} dB | Time: {elapsed_time:.2f} seconds")
    
    def train_noval(self, epochs, model=None):
        # 设置模型设备
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model.to(self.device)
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}\n")
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            train_loss, train_time = self.train_epoch(epoch, self.model)
            
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"Train Loss: {train_loss:.8f} | Time: {train_time:.2f}s")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            mse, psnr, elapsed_time = self.test()
            print(f"Test MSE: {mse:.4f} | PSNR: {psnr:.4f} dB | Time: {elapsed_time:.2f} seconds")
    
            # print weight sum values
            weight_sum = sum(p.sum().item() for p in self.model.parameters() if p.requires_grad)
            print(f"Weight sum: {weight_sum:.4f}")
            # mse, psnr, elapsed_time = model_trainer.test(model_trainer.model)
            


            # # 每5轮保存一次中间模型
            # if epoch % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoints/model_epoch{epoch}.pth')
            #     print(f"Intermediate model saved at epoch {epoch}")
            # if epoch % 5 == 0:
            #     param_norms = {}
            #     for name, param in self.model.named_parameters():
            #         if param.requires_grad and param.grad is not None:
            #             param_norms[name] = param.grad.norm().item()
            #     print(f"参数梯度范数: {param_norms}")
        # # 训练完成后保存最终模型
        # final_model_path = f'checkpoints/model_final_epoch{epochs}.pth'
        # torch.save(self.model.state_dict(), final_model_path)
        # print(f"\nFinal model saved at: {final_model_path}")
        
        # # 使用最终模型进行测试
        # print("\nTraining completed. Running test with final model...")
        mse, psnr, elapsed_time = self.test()
        print(f"Test MSE: {mse:.4f} | PSNR: {psnr:.4f} dB | Time: {elapsed_time:.2f} seconds")