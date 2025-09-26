from datetime import datetime
import os
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
from datasets import Syn_NTIRE, test_data_NTIRE, PairedDenoisingDataset
from torch.utils.data import ConcatDataset
from my_models import UnetGenerator_hardware, UnetGenerator_hardware_pixelshuffle
from collections import defaultdict

sys.argv = [
    'train.py',  # Placeholder script name
    '--dataroot', './datasets/maps',
    '--name', 'denoiser_pix2pix',
    '--model', 'pix2pix',
    '--gpu_ids', '3',  # Set other options here
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


def checkpoint_test_multi(test_loader, model, best, epoch, best_epoch, text_path,
                    denoiser_model, device, curr_epch, optimizer, path,
                    train_loss, scheduler):
    psnr_total = 0.0
    count_total = 0

    psnr_per_noise = defaultdict(float)
    count_per_noise = defaultdict(int)

    denoiser_model.eval()
    with torch.no_grad():
        for index, (noisy, clean, noise_level) in enumerate(test_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            noisy, _ = pad_to_multiple(noisy, multiple=256)
            clean, padding = pad_to_multiple(clean, multiple=256)

            denoised = denoiser_model(noisy)
            denoised = torch.clamp(denoised, 0, 1)

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

    # Print summary
    print(f"[Epoch {epoch}] Average PSNR: {ave_psnr:.2f}")
    print(" | ".join(per_noise_strs))

    # Log new best if found
    if ave_psnr > best:
        best = ave_psnr
        best_epoch = epoch
        print(f" New best ".center(50, "*"))
        checkpoint(curr_epch, train_loss, denoiser_model,
                   optimizer, path, text_path, scheduler, ave_psnr)

    print(f"Curr PSNR: {ave_psnr:.2f} \t Best: {best:.2f} \t Best epoch: {best_epoch} \t Curr LR: {get_lr(optimizer):.6f}")
    return best, best_epoch


def train_gray(epoch, data_loader, device, optimizer,
               scheduler, denoiser_model):
    denoiser_model.train()

    train_loss = 0.0
    # start_time = time.perf_counter()
    for (noise_img, clean_img) in data_loader:

        noise_img = noise_img.to(device)
        clean_img = clean_img.to(device)
        denoised = denoiser_model(noise_img)
        # denoised = torch.clamp(wiener_fildenoisedered, 0, 1)
        im_loss = F.l1_loss(denoised, clean_img)

        train_loss += im_loss.item()
        optimizer.zero_grad()
        im_loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss = train_loss / len(data_loader)

    print(f"Epoch {epoch} \tTotal loss: {train_loss:.4f}")
    # checkpoint(i, loss.item(), wiener, optimizer, path, text_path)
    # checkpoint_noise_predictor(i, loss.item(), model, optimizer, path, text_path)
    return train_loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def checkpoint_noise_predictor(epoch, train_loss, model, optimizer, path, text_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'noise_model_state_dict': noise_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss
            }, path+"/model_epoch_{}.pt".format(epoch))
    print("Epoch saved")
    with open(text_path, 'a') as myfile:
        myfile.write("Epoch: {} \tLoss: {}\n".format(epoch, train_loss))

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

    dataset = Syn_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/train_imgs/', 150, patch_size=256) + Syn_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/DIV2K/', 150, patch_size=256)
    # dataset = Syn_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/train_imgs/', 1, patch_size=256) 
    # test_dataset = test_data_NTIRE('/home/bledc/dataset/denoising_challenge/test_imgs/', 100, patch_size=256) #center crop excluded atm

    # test_dataset = test_data_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/test_imgs/', 100, patch_size=256) #center crop excluded atm
    # test_dataset = PairedDenoisingDataset("/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/BSD/noise5/")
    test_dataset = ConcatDataset([
        PairedDenoisingDataset("/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/BSD/noise5/", noise_level=5),
        PairedDenoisingDataset("/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/BSD/noise20/",noise_level=20),
        PairedDenoisingDataset("/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/BSD/noise40/",noise_level=40)
    ])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64,
        shuffle=True, num_workers=6,
        pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=3)


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)



    model = create_model(TrainOptions().parse())
    denoiser_model = UnetGenerator_hardware(3, 3, 8).to(device)

    # denoiser_model = UnetGenerator_hardware_pixelshuffle(3, 3, 4).to(device)

    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges16_49_15/model_epoch_1100.pt" # hardware changes model path (not trained to completeion bc of scheduling issues)
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges_continuetrianing10_29_55/model_epoch_3060.pt" # hardware changes model path (not trained to completeion bc of scheduling issues)
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges_continuetrianing11_05_02/model_epoch_4920.pt" # hardware changes model path (not trained to completeion bc of scheduling issues)
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges_pixelshuffle_shallow16_03_17/model_epoch_1240.pt"
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges_pixelshuffle15_18_42/model_epoch_9950.pt" # pixel shuffle

    model_path = "/data/clement/models/regular_unet_proper_data_multitest_continue10_06_25/model_epoch_770.pt" # original model again proper dataset
    
    model_dict = torch.load(model_path, map_location=device)
    # model_dict = strip_module_prefix(model_dict['model_state_dict'])
    # # model_dict = remap_state_dict_keys(model_dict, denoiser_model)
    denoiser_model.load_state_dict(model_dict["model_state_dict"])

    # denoiser_model = model.netG  # assuming the generator is the denoiser model
    # remove_batchnorm_layers(denoiser_model)
    # replace_tanh_with_relu(denoiser_model)
    # remove_dropout_layers(denoiser_model)
    learning_rate = 1e-4

    # optimizera = torch.optim.Adam(denoiser_model.parameters(), lr=learning_rate)
    optimizera = torch.optim.AdamW(denoiser_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizera, T_0=5000, T_mult=2)
    optimizer_dict = torch.load(model_path, map_location=device)
    optimizera.load_state_dict(optimizer_dict['optimizer_state_dict'])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizera, T_0=5000, T_mult=2)


    loss = 11
    start_time = time.perf_counter()
    best_epoch = 0
    i = 0
    best = 0
    num_epochs = 10000
    train_loss = 999.0
    print("Commencing training")

    for i in range(0, num_epochs + 1):

        if i % 10 == 0:
            if i ==0:
                experiment_path, current_time = path_config.get_experiment_dir()
                os.makedirs(experiment_path, exist_ok=True)
                text_path = f"{experiment_path}/{current_time}.txt"
                with open(text_path, 'w') as txt_data:
                    txt_data.write("Training Log\n")
            best, best_epoch = checkpoint_test_multi(test_loader, model, best, i, best_epoch, text_path, denoiser_model,
                                               device, i, optimizera, experiment_path, train_loss, scheduler)

        train_loss = train_gray(i, data_loader, device, optimizera,
                            scheduler, denoiser_model)
        # scheduler.step()
