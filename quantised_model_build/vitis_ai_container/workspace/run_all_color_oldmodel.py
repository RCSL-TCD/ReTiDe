# ----------------------------------------------------------------------------
# Emerald Video Denoise Accelerator
# Benchmarking script for evaluating model performance
# In this script, we benchmark the 2 models with pixel shuffle and without pixel shuffle.
# ----------------------------------------------------------------------------
import os
import cv2
import numpy as np
from pathlib import Path
import torch
import shutil
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import xir
import vart
import time

from typing import List
# ----------------------------------------------------------------------------
from models.models_nndct_0812 import UnetGenerator_hardware_nndct as Unet
# from utils.dataloader import get_dataloader
from utils.preprocessing import process_images, process_images_grayscale, crop_and_save_images, crop_and_save_images_color
from utils.dataloader import DenoiseDatasetGrayScale, get_grayscale_loaders, DenoiseDatasetColor, get_multi_noise_color_loaders
from utils.testmodel import print_test_results, test_denoising_model_grayscale
from utils.model_trainer import Trainer, calculate_raw_metrics
from utils.quantize import quantize, quantize_qat, quantize_qat_gray, quantize_qat_color
#from utils.performance import measure_model_performance, print_performance_results
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# model definition
# ----------------------------------------------------------------------------
# float32 model
# unet_f32_weight_path = 'models/unet_f32_0806_color_oldtorch.pt'
unet_f32_weight_path = 'models/unet_f32_0814_color_oldtorch.pt'

# ----------------------------------------------------------------------------
# test set definition
# ----------------------------------------------------------------------------

# NTIRE2
dataset_ntire2_path = 'dataset/train_NTIRE2_color'
# BSD68C
dataset_bsd68c_path = 'dataset/test_BSD68C'

# ----------------------------------------------------------------------------
# hyper-paras
# ----------------------------------------------------------------------------
batch_size = 32
input_nc = 3
output_nc = 3
num_downs = 8


spliter = '-------------------------------------------------------'


def test_load(datapath, unet_f32_weight_path):
    print('preprocessing images')
    print(spliter)
    output_patch_path = f"{datapath}_patches"
    process_images(datapath)
    clean_src = os.path.join(datapath, 'clean')
    clean_dst = os.path.join(output_patch_path, 'clean')
    crop_and_save_images_color(clean_src, clean_dst, 'clean')
    noise_levels = [d for d in os.listdir(datapath) 
                    if d.startswith('noise') and os.path.isdir(os.path.join(datapath, d))]
    for level in noise_levels:
        noise_value = level.replace('noise', '')
        src_dir = os.path.join(datapath, level)
        dst_dir = os.path.join(output_patch_path, f'noisy{noise_value}')
        crop_and_save_images_color(src_dir, dst_dir, f'noisy')

    print('initializing the model')
    print(spliter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'running with {device}')
    model = Unet(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs) 
    model.load_state_dict(torch.load(unet_f32_weight_path))
    model.to(device)
    print('model loaded')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")


    print('excuting the denoising')
    print(spliter)
    for level in noise_levels:
        noise_value = level.replace('noise', '')
        print(f"\nProcessing noise level: {noise_value}")
        
        noisy_dir = os.path.join(output_patch_path, f'noisy{noise_value}')
        clean_dir = os.path.join(output_patch_path, 'clean')
        dataset = DenoiseDatasetColor(noisy_dir, clean_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        output_dir = os.path.join(output_patch_path, f'should_denoise{noise_value}')

        test_denoising_model_grayscale(
            model=model,
            dataloader=dataloader,
            device=device,
            output_path=output_dir
        )
    print(f"Results saved to: {output_dir}")

def PTQ(model_path):
    print('Starting PTQ quantization')
    print(spliter)
    train_loader, val_loader, test_loader = get_multi_noise_color_loaders(dataset_ntire2_path, 
                                                                  batch_size=batch_size, 
                                                                  val_ratio=0.1, 
                                                                  test_ratio=0.3)
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches, Test loader: {len(test_loader)} batches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    
    model = Unet(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    benchmark_trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    raw_mse, raw_psnr = calculate_raw_metrics(test_loader, device=device)
    print(f"Raw MSE: {raw_mse:.4f}, Raw PSNR: {raw_psnr:.2f} dB")
    mse, psnr, test_time = benchmark_trainer.test()
    f32_mse, f32_psnr = mse, psnr
    print(f"Float32 MSE: {mse:.4f}, PSNR: {psnr:.2f} dB, Time: {test_time:.2f} seconds")

    print(spliter)
    print('PTQ quantization')
    mse, psnr, elapsed_time, quantized_model = quantize('calib',
                                                         batch_size,
                                                           model,
                                                             benchmark_trainer,
                                                               size = [3, 256, 256])
    
    # force the batch size to 1 from the dataset level
    train_loader_e, val_loader_e, test_loader_e = get_multi_noise_color_loaders(dataset_ntire2_path, 
                                                                batch_size=1, 
                                                                val_ratio=0.1, 
                                                                test_ratio=0.3)
    trainer_e = Trainer(quantized_model, train_loader_e, val_loader_e, test_loader_e, device)
    mse, psnr, elapsed_time, quantized_model = quantize('test',
                                                        1,
                                                        model,
                                                            trainer_e,
                                                            size = [3, 256, 256])
    print(f"PTQ Quantized Model Metrics:\nMSE: {mse:.4f}, PSNR: {psnr:.2f} dB, Time: {elapsed_time:.2f} seconds")
    return mse, psnr, quantized_model, f32_mse, f32_psnr, raw_mse, raw_psnr

def QAT(model_path, qat_epochs=10):
    print('Starting QAT quantization')
    print(spliter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')
    model = Unet(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs)
    model.load_state_dict(torch.load(model_path))
    test_loss, psnr, final_quantized_model = quantize_qat_color(model, dataset_ntire2_path, batch_size=32, qat_epochs=qat_epochs)

    return test_loss, psnr, final_quantized_model


if __name__ == "__main__":
    print('start testing')
    test_load(dataset_bsd68c_path, unet_f32_weight_path)
    #test_load(dataset_ntire2_path, unet_f32_weight_path)
    mse_p, psnr_p, quantized_model_p, f32_mse, f32_psnr, raw_mse, raw_psnr = PTQ(unet_f32_weight_path)
    #mse_q, psnr_q, quantized_model_q = QAT(unet_f32_weight_path, qat_epochs=30)