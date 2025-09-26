import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
# custom imports
import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel, QatProcessor
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset, DataLoader
from utils.testmodel import print_test_results, test_denoising_model_grayscale
from torchvision import transforms
from PIL import Image
from utils.model_trainer import Trainer
from utils.dataloader import get_grayscale_loaders, DenoiseDatasetGrayScale, get_multi_noise_loaders, get_multi_noise_color_loaders, DenoiseDatasetColor
import torch.optim as optim
import math
DIVIDER = '-----------------------------------------'
quant_model = 'build/quant_model'


def quantize(quant_mode, batchsize, model, model_trainer, size=[3, 256, 256], qat_epochs=30):


    if quant_mode == 'test':
        batchsize = 1

    channel_i, height_i, width_i = size
    rand_in = torch.randn([batchsize, channel_i, height_i, width_i])
    quantizer = torch_quantizer(quant_mode, model, rand_in, output_dir=quant_model)
    quantized_model = quantizer.quant_model

    mse, psnr, elapsed_time = model_trainer.test(quantized_model)
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    elif quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
 
    return mse, psnr, elapsed_time, quantized_model

def quantize_qat(batchsize, model, model_trainer, size=[3, 256, 256], qat_epochs=10, trainer2=None):
    channel_i, height_i, width_i = size
    rand_in = torch.randn([batchsize, channel_i, height_i, width_i])
    device = torch.device('cpu')    # QAT only support CPU
    model.to(device)
    qat_processor = QatProcessor(model, rand_in, 8, device = device)
    quantized_model = qat_processor.trainable_model(allow_reused_module=False)
    model_trainer.model = quantized_model
    model_trainer.train(qat_epochs, quantized_model)
    quantized_model = model_trainer.model
    test_loader = trainer2.test_loader
    deployable_model = qat_processor.to_deployable(quantized_model, "qat_results")
    deployable_model.eval()
    trainer2.model = deployable_model
    deployable_model.to(device)
   
    mse, psnr, elapsed_time = trainer2.test(deployable_model)
    deployable_model.to(device)


    for img, _ in test_loader:
        img = img.to(device)
        with torch.no_grad():
            output = deployable_model(img)

    qat_processor.export_xmodel(output_dir="build/quant_qat", deploy_check=False)
    return mse, psnr, elapsed_time, deployable_model

def quantize_qat_color(float_model, data_path, batch_size=16, qat_epochs=10, lr=1e-8):
    # Load the float model
    model = float_model
    device = torch.device('cpu')  # QAT only supports CPU
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Create a dummy input for QAT
    rand_in = torch.randn([batch_size, 3, 256, 256])
    #train_loader, val_loader, test_loader = get_grayscale_loaders(data_path, batch_size=batch_size, val_ratio=0.1, test_ratio=0.3)
    train_loader, val_loader, test_loader = get_multi_noise_color_loaders(data_path, batch_size=batch_size, val_ratio=0.1, test_ratio=0.3) # 0.1 0.2 works
    # get one pair of train data, save to /output
    for noisy_img, clean_img in train_loader:
        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)
        os.makedirs('output', exist_ok=True)
        torchvision.utils.save_image(noisy_img, 'output/noisy_img.png')
        torchvision.utils.save_image(clean_img, 'output/clean_img.png')
        break
    #rand_in = torch.randn([batch_size, 1, 256, 256])
    qat_processor = QatProcessor(model, rand_in, 8, device = device)
    quantized_model = qat_processor.trainable_model(allow_reused_module=False)

        # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 适用于图像去噪任务
    optimizer = optim.Adam(quantized_model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantized_model.to(device)
    # QAT训练循环
    for epoch in range(qat_epochs):
        # 训练阶段
        quantized_model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{qat_epochs} [Train]"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = quantized_model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        quantized_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{qat_epochs} [Val]"):
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                outputs = quantized_model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item() * noisy_imgs.size(0)
        
        val_loss /= len(val_loader.dataset)
        psnr = 10 * math.log10(1 / val_loss) if val_loss > 0 else float('inf')


        print(f"Epoch {epoch+1}/{qat_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {psnr:.2f} dB")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(quantized_model.state_dict(), 'best_qat_model.pth')
    

    device = torch.device('cpu')
    rand_in = torch.randn([1, 3, 256, 256])
    qat_processor = QatProcessor(model, rand_in, 8, device = device)
    quantized_model = qat_processor.trainable_model(allow_reused_module=False)
    # 加载最佳模型并转换为最终量化模型
    quantized_model.load_state_dict(torch.load('best_qat_model.pth', map_location=device))
    final_quantized_model = qat_processor.to_deployable(quantized_model, "qat_results")
    

    final_quantized_model.to(device)
    #train_loader, val_loader, test_loader = get_grayscale_loaders(data_path, batch_size=1, val_ratio=0.1, test_ratio=0.3)
    train_loader, val_loader, test_loader = get_multi_noise_color_loaders(data_path, batch_size=1, val_ratio=0.1, test_ratio=0.3) # 0.2 0.2 works
    # ** batch size here must be 1 to build the xmodel **
    # 在测试集上评估最终模型
    final_quantized_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in tqdm(test_loader, desc="Testing Final Model"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            outputs = final_quantized_model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            test_loss += loss.item() * noisy_imgs.size(0)
    
    test_loss /= len(test_loader.dataset)
    psnr = 10 * math.log10(1 / test_loss) if test_loss > 0 else float('inf')
    print(f"Final Quantized Model Test Loss: {test_loss:.6f} | PSNR: {psnr:.2f} dB")


    # further test
    data_path = 'dataset/test_BSD68C'
    noise_levels = [d for d in os.listdir(data_path) 
            if d.startswith('noise') and os.path.isdir(os.path.join(data_path, d))]
    output_patch_path = f"{data_path}_patches"

    for level in noise_levels:
        noise_value = level.replace('noise', '')
        print(f"\nProcessing noise level: {noise_value}")
        
        noisy_dir = os.path.join(output_patch_path, f'noisy{noise_value}')
        clean_dir = os.path.join(output_patch_path, 'clean')
        dataset = DenoiseDatasetColor(noisy_dir, clean_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        output_dir = os.path.join(output_patch_path, f'denoise_test_qat{noise_value}')

        # if the output directory does not exist, create it
        os.makedirs(output_dir, exist_ok=True)

        print(f"Testing quantized model on {len(dataloader)} batches of size {batch_size}...")

        test_denoising_model_grayscale(
            model=final_quantized_model,
            dataloader=dataloader,
            device=device,
            output_path=output_dir
        )
        print(f"Results saved to: {output_dir}")

    print('exporting final quantized model to xmodel...')
    # 保存示例输出图像
    sample_input = next(iter(test_loader))[0][:1].to(device)
    with torch.no_grad():
        sample_output = final_quantized_model(sample_input)
    os.makedirs('output', exist_ok=True)
    torchvision.utils.save_image(sample_input, 'output/test_input.png')
    torchvision.utils.save_image(sample_output, 'output/test_output.png')
    # must do one inference to ensure the model is working
    for img, _ in test_loader:
        img = img.to(device)
        with torch.no_grad():
            output = final_quantized_model(img)

    qat_processor.export_xmodel(output_dir="build/quant_qat", deploy_check=False)
    return test_loss, psnr, final_quantized_model


def quantize_qat_gray(float_model, data_path, batch_size=16, qat_epochs=10, lr=1e-8):
    # Load the float model
    model = float_model
    device = torch.device('cpu')  # QAT only supports CPU
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Create a dummy input for QAT
    rand_in = torch.randn([batch_size, 1, 256, 256])
    #train_loader, val_loader, test_loader = get_grayscale_loaders(data_path, batch_size=batch_size, val_ratio=0.1, test_ratio=0.3)
    train_loader, val_loader, test_loader = get_multi_noise_loaders(data_path, batch_size=batch_size, val_ratio=0.1, test_ratio=0.3) # 0.1 0.2 works
    # get one pair of train data, save to /output
    for noisy_img, clean_img in train_loader:
        noisy_img = noisy_img.to(device)
        clean_img = clean_img.to(device)
        os.makedirs('output', exist_ok=True)
        torchvision.utils.save_image(noisy_img, 'output/noisy_img.png')
        torchvision.utils.save_image(clean_img, 'output/clean_img.png')
        break
    #rand_in = torch.randn([batch_size, 1, 256, 256])
    qat_processor = QatProcessor(model, rand_in, 8, device = device)
    quantized_model = qat_processor.trainable_model(allow_reused_module=False)

        # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 适用于图像去噪任务
    optimizer = optim.Adam(quantized_model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quantized_model.to(device)
    # QAT训练循环
    for epoch in range(qat_epochs):
        # 训练阶段
        quantized_model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{qat_epochs} [Train]"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = quantized_model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        quantized_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{qat_epochs} [Val]"):
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                outputs = quantized_model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item() * noisy_imgs.size(0)
        
        val_loss /= len(val_loader.dataset)
        psnr = 10 * math.log10(1 / val_loss) if val_loss > 0 else float('inf')


        print(f"Epoch {epoch+1}/{qat_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PSNR: {psnr:.2f} dB")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(quantized_model.state_dict(), 'best_qat_model.pth')
    

    device = torch.device('cpu')
    rand_in = torch.randn([1, 1, 256, 256])
    qat_processor = QatProcessor(model, rand_in, 8, device = device)
    quantized_model = qat_processor.trainable_model(allow_reused_module=False)
    # 加载最佳模型并转换为最终量化模型
    quantized_model.load_state_dict(torch.load('best_qat_model.pth', map_location=device))
    final_quantized_model = qat_processor.to_deployable(quantized_model, "qat_results")
    

    final_quantized_model.to(device)
    #train_loader, val_loader, test_loader = get_grayscale_loaders(data_path, batch_size=1, val_ratio=0.1, test_ratio=0.3)
    train_loader, val_loader, test_loader = get_multi_noise_loaders(data_path, batch_size=1, val_ratio=0.1, test_ratio=0.3) # 0.2 0.2 works
    # ** batch size here must be 1 to build the xmodel **
    # 在测试集上评估最终模型
    final_quantized_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in tqdm(test_loader, desc="Testing Final Model"):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            outputs = final_quantized_model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            test_loss += loss.item() * noisy_imgs.size(0)
    
    test_loss /= len(test_loader.dataset)
    psnr = 10 * math.log10(1 / test_loss) if test_loss > 0 else float('inf')
    print(f"Final Quantized Model Test Loss: {test_loss:.6f} | PSNR: {psnr:.2f} dB")


    # further test
    data_path = 'dataset/test_BSD68'
    noise_levels = [d for d in os.listdir(data_path) 
            if d.startswith('noise') and os.path.isdir(os.path.join(data_path, d))]
    output_patch_path = f"{data_path}_patches"

    for level in noise_levels:
        noise_value = level.replace('noise', '')
        print(f"\nProcessing noise level: {noise_value}")
        
        noisy_dir = os.path.join(output_patch_path, f'noisy{noise_value}')
        clean_dir = os.path.join(output_patch_path, 'clean')
        dataset = DenoiseDatasetGrayScale(noisy_dir, clean_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        output_dir = os.path.join(output_patch_path, f'denoise_test_qat{noise_value}')

        # if the output directory does not exist, create it
        os.makedirs(output_dir, exist_ok=True)

        print(f"Testing quantized model on {len(dataloader)} batches of size {batch_size}...")

        test_denoising_model_grayscale(
            model=final_quantized_model,
            dataloader=dataloader,
            device=device,
            output_path=output_dir
        )
        print(f"Results saved to: {output_dir}")

    print('exporting final quantized model to xmodel...')
    # 保存示例输出图像
    sample_input = next(iter(test_loader))[0][:1].to(device)
    with torch.no_grad():
        sample_output = final_quantized_model(sample_input)
    os.makedirs('output', exist_ok=True)
    torchvision.utils.save_image(sample_input, 'output/test_input.png')
    torchvision.utils.save_image(sample_output, 'output/test_output.png')
    # must do one inference to ensure the model is working
    for img, _ in test_loader:
        img = img.to(device)
        with torch.no_grad():
            output = final_quantized_model(img)

    qat_processor.export_xmodel(output_dir="build/quant_qat", deploy_check=False)
    return test_loss, psnr, final_quantized_model

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return

if __name__ == '__main__':
    run_main()