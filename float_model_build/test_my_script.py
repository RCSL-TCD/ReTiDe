from ast import mod
from datetime import datetime
import os
from numpy import save
import torch
from torchvision.utils import save_image
from torch.nn import functional as F
from math import log10
from torch import nn
import sys
from options.train_options import TrainOptions
import path_config
from models import create_model
from datasets import  test_data_NTIRE
from collections import OrderedDict
from my_models import UnetGenerator, UnetGenerator_hardware, UnetGenerator_hardware_pixelshuffle
sys.argv = [
    'train.py',  # Placeholder script name
    '--dataroot', './datasets/maps',
    '--name', 'denoiser_pix2pix',
    '--model', 'pix2pix',
    '--gpu_ids', '1',  # Set other options here
    '--n_epochs', '100',
    '--n_epochs_decay', '100'
]

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

def pad_to_multiple(x, multiple=256):
    _, _, h, w = x.size()
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad = (0, pad_w, 0, pad_h)  # pad (left, right, top, bottom)
    return nn.functional.pad(x, pad, mode='reflect'), pad

def crop_to_original(x, padding):
    pad_left, pad_right, pad_top, pad_bottom = padding
    _, _, h, w = x.shape
    return x[..., pad_top:h - pad_bottom, pad_left:w - pad_right]


def checkpoint_test(test_loader, denoiser_model,  device, 
                    path):
    psnr = 0.0
    ave_psnr = 0.0
    denoiser_model.eval()
    with torch.no_grad():
        for index, (noise_img, clean_img) in enumerate(test_loader):
            noise_img = noise_img.to(device)
            clean_img = clean_img.to(device)
            # Pad images to multiple of 256
            noise_img, pad = pad_to_multiple(noise_img, 256)
            denoised = denoiser_model(noise_img)
            denoised = torch.clamp(denoised, 0, 1)
            # Remove padding
            denoised = denoised[:, :, :clean_img.size(2), :clean_img.size(3)]
            noise_img = crop_to_original(noise_img, pad)
            # combined = torch.cat((noise_img, denoised, clean_img), dim=0)
            # save_image(combined, os.path.join(path, f"test_{index}.png"))
            save_image(denoised, os.path.join(path, f"{index}_denoised.png"))
            # save_image(clean_img, os.path.join(path, f"{index}_clean.png"))
            # save_image(noise_img, os.path.join(path, f"{index}_noise.png"))
            psnr = test_psnr(clean_img, denoised)
            print(f"Processed image {index + 1}/{len(test_loader)}, PSNR: {psnr:.2f} dB")
            ave_psnr += psnr
        ave_psnr = ave_psnr / len(test_loader)
        
            
        print(f"AVE PSNR: {ave_psnr: .2f}")
        return 


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


if __name__ == '__main__':
    experiment_name = path_config.experiment_name
    
    torch.multiprocessing.set_start_method('spawn')
    torch.set_printoptions(linewidth=120)
    now = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    current_time = now.strftime("%H_%M_%S")


    test_dataset = test_data_NTIRE('/home/bledc@ad.mee.tcd.ie/data/clement/images/datasets/denoising_challenge/test_imgs/', 100, patch_size=256) #center crop excluded atm
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=8,
        pin_memory=True, drop_last=False)

    print(device)

    
    # model = create_model(TrainOptions().parse())
    # denoiser_model = model.netG  # assuming the generator is the denoiser model
    # remove_batchnorm_layers(denoiser_model)
    # replace_tanh_with_relu(denoiser_model)
    # remove_dropout_layers(denoiser_model)
    
    # denoiser_model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8).to(device)
    
    denoiser_model = UnetGenerator_hardware(3, 3, 8).to(device)
    denoiser_model = UnetGenerator_hardware_pixelshuffle(3, 3, 4).to(device)
    
    # Need to modify the dicts keys to match the new model
    # model_path = "/data/clement/models/training_pix2pix_denoiser14_56_27/model_epoch_4920.pt" # default model path
    
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges16_49_15/model_epoch_1210.pt" # hardware changes model path (not trained to completeion bc of scheduling issues)
    
    # model_path = "/data/clement/models/training_pix2pix_denoiser_hardwarechanges_pixelshuffle15_18_42/model_epoch_9950.pt" # pixel shuffle
    
    # model_dict = torch.load(model_path, map_location=device)
    # model_dict = strip_module_prefix(model_dict['model_state_dict'])
    # model_dict = remap_state_dict_keys(model_dict, denoiser_model)
    
    
    # denoiser_model.load_state_dict(model_dict)
    # subddir = "MyModels"
    # os.makedirs(subddir, exist_ok=True)
    # model_save_path = os.path.join(subddir, f"{experiment_name}_denoiser.pth")
    # torch.save(denoiser_model.state_dict(), model_save_path)

    
    experiment_path, current_time = path_config.get_experiment_dir()
    benchmark_path = path_config.benchmark_save_path
    
    
    experiment_path = os.path.join(benchmark_path, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    print(f"\nSAVING OUTPUT AT: {experiment_path}")
    checkpoint_test(test_loader, denoiser_model, 
                                        device, experiment_path)
