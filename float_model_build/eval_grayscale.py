import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime

import torch
from torchvision.utils import save_image
from torch.nn import functional as F
from math import log10
import math
import time
from collections import OrderedDict
from torch import nn
import torchvision
from datasets import Syn_NTIRE
import numpy as np
import sys
from options.train_options import TrainOptions
import path_config

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from datasets import Syn_NTIRE, test_data_NTIRE, PairedDenoisingDataset, Syn_NTIRE_improved, PairedDenoisingDataset_eval
from torch.utils.data import ConcatDataset
from my_models import UnetGenerator_hardware, UnetGenerator_hardware_pixelshuffle, UnetGenerator_3stage
from collections import defaultdict


sys.argv = [
    'train.py',  # Placeholder script name
    '--dataroot', './datasets/maps',
    '--name', 'denoiser_pix2pix',
    '--model', 'pix2pix',
    '--gpu_ids', '0',  # Set other options here
    '--n_epochs', '100',
    '--n_epochs_decay', '100'
]

def pad_to_multiple(img_tensor, multiple=16):
    """Pad a 4D tensor (B, C, H, W) so that H and W are divisible by `multiple`."""
    _, _, h, w = img_tensor.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)

    return torch.nn.functional.pad(img_tensor, padding, mode='reflect'), padding


experiment_name = "Mar16_gaussian_competition_largerwienernet_64x64"
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def test_psnr(ground_truth, denoised):
    criterion = nn.MSELoss()
    avg_psnr = 0
    with torch.no_grad():
        for gt, densd in zip(ground_truth, denoised):
            mse = criterion(densd, gt)
            psnr = 10 * log10(1 / mse.item())
            torch.cuda.empty_cache()
            avg_psnr += psnr

    return avg_psnr / len(ground_truth)

def replace_tanh_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Tanh):
            print(f"Replacing Tanh at {name} with ReLU")
            setattr(module, name, nn.ReLU())
        else:
            replace_tanh_with_relu(child)




def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

def remove_batchnorm_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            print(f"Removing BatchNorm: {name}")
            setattr(model, name, nn.Identity())
        else:
            remove_batchnorm_layers(module)

def remove_dropout_layers(model):
    """
    Recursively removes all nn.Dropout layers from a model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            print(f"Removing Dropout layer: {name}")
            setattr(model, name, nn.Identity())
        else:
            remove_dropout_layers(module)

def replace_bn(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            print('replaced: ', name, attr_str)
            new_bn = torch.nn.GroupNorm(32, target_attr.num_features)
            setattr(module, attr_str, new_bn)

    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)


def load_ckpt(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


def get_std(noise_imgs, clean_img):
    num_samples = noise_imgs.shape[1]
    noise_imgs = noise_imgs.swapaxes(0,1)
    noise_only = torch.ones(noise_imgs.shape).cuda()
    for i, noise_img in enumerate(noise_imgs):
        noise_only[i] = noise_img - clean_img
    noise_square = noise_only ** 2
    noise_sum = torch.sum(noise_square, 0)
    noise_mean = noise_sum / num_samples
    noise_std = torch.sqrt(noise_mean)
    return noise_std


def get_psd(noise_imgs, clean_img, std_mean):
    num_samples = noise_imgs.shape[1]
    noise_imgs = noise_imgs.swapaxes(0, 1)
    noise_norm = torch.ones(noise_imgs.shape).cuda()
    for i, noise_img in enumerate(noise_imgs):
        noise_norm[i] = (noise_img - clean_img) / (std_mean + 0.0001)
    noise_norm = noise_norm / num_samples
    return noise_norm

# checkpoint(curr_epch, train_loss, denoiser_model,
#                        optimizer, path, text_path, scheduler, ave_psnr)
def checkpoint(epoch, train_loss, model, optimizer, path, text_path, scheduler, final_psnr):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss
            }, path + "/model_epoch_{}.pt".format(epoch))
    with open(text_path, 'a') as myfile:
        myfile.write("Epoch: {} \tLoss: {:.6f} \tPSNR: {:.2f} dB\n".format(epoch, train_loss, final_psnr))



def checkpoint_test(test_loader, model, best, epoch, best_epoch, text_path,denoiser_model, device,
                    curr_epch, optimizer, path, train_loss, scheduler):
    psnr = 0.0
    denoiser_model.eval()
    with torch.no_grad():
        for index, (noise_img, clean_img, noise_level) in enumerate(test_loader):
            noise_img = noise_img.to(device)
            clean_img = clean_img.to(device)
            denoised = denoiser_model(noise_img)
            denoised = torch.clamp(denoised, 0, 1)
            psnr += test_psnr(clean_img, denoised)

        ave_psnr = psnr / len(test_loader)

        if ave_psnr > best:
            best = ave_psnr
            best_epoch = i
            print(f" New best ".center(50, "*"))
            checkpoint(curr_epch, train_loss, denoiser_model,
                       optimizer, path, text_path, scheduler, ave_psnr)

        print(f"Curr PSNR: {ave_psnr: .2f} \t Best: {best: .2f}\t Best epoch: {best_epoch}\t Curr LR: {get_lr(optimizer): .6f}")
        return best, best_epoch


def checkpoint_test_multi(test_loader, denoiser_model, device):
    psnr_total = 0.0
    count_total = 0

    psnr_per_noise = defaultdict(float)
    count_per_noise = defaultdict(int)

    denoiser_model.eval()
    with torch.no_grad():
        for index, (noisy, clean, noise_level) in enumerate(test_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Get original shape before padding
            _, _, h_orig, w_orig = clean.shape

            padded_noisy, padding = pad_to_multiple(noisy, multiple=64)
            
            denoised_padded = denoiser_model(padded_noisy)
            
            # Unpack padding values
            pad_left, pad_right, pad_top, pad_bottom = padding
            
            # Crop the denoised image back to the original size
            denoised = denoised_padded[:, :, pad_top:pad_top + h_orig, pad_left:pad_left + w_orig]
            
            denoised = torch.clamp(denoised, 0, 1)

            # Now 'clean' is the original unpadded tensor and 'denoised' is cropped to match its size
            psnr_value = test_psnr(clean, denoised)
            psnr_total += psnr_value
            count_total += 1

            sigma = noise_level.item()
            psnr_per_noise[sigma] += psnr_value
            count_per_noise[sigma] += 1

    # Compute average PSNR overall
    ave_psnr = psnr_total / count_total

    # Compute per-noise PSNR averages
    per_noise_strs = []
    for sigma in sorted(psnr_per_noise.keys()):
        avg_sigma_psnr = psnr_per_noise[sigma] / count_per_noise[sigma]
        per_noise_strs.append(f"σ={int(sigma)} PSNR={avg_sigma_psnr:.2f}")

    print(f"Curr PSNR: {ave_psnr:.2f} | {' '.join(per_noise_strs)}")
    return


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def strip_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def remap_state_dict_keys(old_state_dict, new_model):
    # 1. Extract keys in order
    old_keys = list(old_state_dict.keys())
    new_keys = list(new_model.state_dict().keys())

    if len(old_keys) != len(new_keys):
        print(f"⚠️  Key count mismatch: {len(old_keys)} (old) vs {len(new_keys)} (new)")
        min_len = min(len(old_keys), len(new_keys))
        print("Only mapping the first", min_len, "keys")
    else:
        min_len = len(new_keys)

    # 2. Map in order
    new_state_dict = OrderedDict()
    for i in range(min_len):
        new_state_dict[new_keys[i]] = old_state_dict[old_keys[i]]

    return new_state_dict

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('ConvBlock') != -1:
        pass
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights_ones(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_printoptions(linewidth=120)
    now = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    current_time = now.strftime("%H_%M_%S")

    test_dataset = ConcatDataset([
        PairedDenoisingDataset_eval("/home/bledc/Pictures/urban100/noise50_gray", noise_level=50),
        # PairedDenoisingDataset("/home/bledc/Pictures/CBSD68-dataset/CBSD68/grayscale_noise25", noise_level=25),
        # PairedDenoisingDataset("/home/bledc/Pictures/CBSD68-dataset/CBSD68/grayscale_noise50", noise_level=50)
    ])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    # denoiser_model = UnetGenerator_hardware(1, 1, 6).to(device)
    denoiser_model = UnetGenerator_3stage(1,1).to(device)
    
    
    # model_path = "/data/clement/models/hw_denoiser_grayscale_15_07_34/model_epoch_1580.pt"
    # model_path = "/data/clement/models/hw_denoiser_grayscale_continue21_10_58/model_epoch_9940.pt"
    # model_path = "/data/clement/models/hw_denoiser_grayscale_256-6downsampleA_Continue18_34_16/model_epoch_9790.pt"
    # model_path = "/data/clement/models/hw_denoiser_grayscale_3stage_addition_resblocks00_15_34/model_epoch_2360.pt"
    
    model_path = "/data/clement/models/hw_denoiser_grayscale_3stage_addition_resblocks00_15_34/model_epoch_2360.pt"
    model_dict = torch.load(model_path, map_location=device)
    denoiser_model.load_state_dict(model_dict["model_state_dict"])


    checkpoint_test_multi(test_loader, denoiser_model, device)
