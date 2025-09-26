import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import glob
from torch.utils.data import Dataset
from torchvision.utils import save_image
# from skimage import color, io
import cv2

def read_img(filename, grayscale=False):
    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filename)
    if img is None or isinstance(img, str):
        print("invalid img")
        print(filename)
        return "None"
    if img.ndim < 3:
        # print("single dim img")
        img = np.expand_dims(img, 2)
        img = img[:,:,::-1] / 255.0
    else:
        img = img[:,:,::-1] / 255.0

    img = np.array(img).astype('float32')

    return img

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).astype('float32')

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).astype('float32')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_patch(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.transpose(imgs[i], (1, 0, 2))

    return imgs

def get_patch(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(imgs)):
            imgs[i] = np.transpose(imgs[i], (1, 0, 2))

    return imgs

def get_patch_multiple_noisy(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0
    noisy = imgs[-1]
    imgs.pop()
    noisy_list = [*noisy]
    new_imgs_list = imgs + noisy_list
    for i in range(len(new_imgs_list)):
        new_imgs_list[i] = new_imgs_list[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(new_imgs_list)):
            new_imgs_list[i] = np.flip(new_imgs_list[i], axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(new_imgs_list)):
            new_imgs_list[i] = np.flip(new_imgs_list[i], axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        for i in range(len(new_imgs_list)):
            new_imgs_list[i] = np.transpose(new_imgs_list[i], (1, 0, 2))

    return new_imgs_list

class test_my_mixed_set(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, noisemap=False):
        self.patch_size = patch_size
        folder_clean = root_dir
        self.clean_fns = glob.glob(folder_clean + '/*GT*')
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(patch_size),
            ])

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx]
        noisy_fn = clean_fn.replace("GT", "NOISY")
        clean_img = read_img(clean_fn)
        noisy_img = read_img(noisy_fn)
        clean_img = self.transforms(clean_img)
        noisy_img = self.transforms(noisy_img)
        return noisy_img, clean_img

def test_patch(imgs, patch_size):
    H = imgs[0].shape[0]
    W = imgs[0].shape[1]

    ps_temp = min(H, W, patch_size)

    xx = np.random.randint(0, W-ps_temp) if W > ps_temp else 0
    yy = np.random.randint(0, H-ps_temp) if H > ps_temp else 0

    for i in range(len(imgs)):
        imgs[i] = imgs[i][yy:yy+ps_temp, xx:xx+ps_temp, :]

    return imgs

class Real(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])

        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

        if self.patch_size > 0:
            [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)

        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.zeros((3, self.patch_size, self.patch_size)), np.zeros((3, self.patch_size, self.patch_size))
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img)

class Real_nonoisemaps(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size

        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])

        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

        if self.patch_size > 0:
            [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)

        return hwc_to_chw(noise_img), hwc_to_chw(clean_img)

class Syn_noisemaps(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        # self.transforms = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=0.5, hue=0.35)])
        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        # sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB')) / 15
        # gaussian_img = read_img(clean_fn.replace('GT_SRGB', 'rand'))
        if self.patch_size > 0:
            # [clean_img, noise_img, sigma_img] = get_patch([clean_img, noise_img, sigma_img], self.patch_size)
            [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)
            # [clean_img, noise_img, sigma_img, gaussian_img] = get_patch([clean_img, noise_img, sigma_img, gaussian_img], self.patch_size)
        noise_img = hwc_to_chw(noise_img)
        clean_img = hwc_to_chw(clean_img)
        # sigma_img = hwc_to_chw(sigma_img)
        # gaussian_img = hwc_to_chw(gaussian_img)
        # return noise_img, clean_img, sigma_img, np.ones((3, self.patch_size, self.patch_size))
        # return noise_img, clean_img, sigma_img, gaussian_img
        return noise_img, clean_img


class Syn_multiple_noisy(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        # self.transforms = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=0.5, hue=0.35)])
        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        noise_imgs = np.repeat(np.expand_dims(np.ones(np.shape(clean_img)), 0), 12, 0)
        # noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY0_SRGB'))
        for i in range(12):
            noise_imgs[i] = read_img(clean_fn.replace('GT_SRGB', f'NOISY{i}_SRGB'))

        if self.patch_size > 0:
            patches = get_patch_multiple_noisy([clean_img, noise_imgs], self.patch_size)
        clean_img = hwc_to_chw(patches[0])
        noisy_ims = np.array(patches[1:])
        noisy_ims_new = np.ones(noisy_ims.shape)
        noisy_ims_new = np.moveaxis(noisy_ims_new, [3], [1])
        for i, image in enumerate(noisy_ims):
            noisy_ims_new[i] = hwc_to_chw(image)

        return noisy_ims_new, clean_img

def read_img_np(filename):
    img = np.load(filename)
    if img is None or isinstance(img, str):
        print("invalid img")
        print(filename)
        return "None"
    img = img[:,:,::-1] / 255.0

    img = np.array(img).astype('float32')
    return img


class Syn_NTIRE(Dataset):
    def __init__(self, root_dir, sample_num, numpy_flag=False, patch_size=128):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((patch_size, patch_size))
            # transforms.CenterCrop((patch_size, patch_size)),
            ])
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_*')
            clean_imgs.sort()

            for ind, clean_img in enumerate(clean_imgs):
                self.clean_fns[ind % sample_num].append(clean_img)
        self.patch_size = patch_size
        self.numpy_flag = numpy_flag

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        # random.seed(seed) # apply this seed to img transforms
        # torch.manual_seed(seed)
        # random_crop = transforms.RandomCrop((self.patch_size, self.patch_size))
        if self.numpy_flag:
            clean_img = read_img_np(clean_fn)
            noise_img = read_img_np(clean_fn.replace('GT_', 'NOISY_'))
        else:
            clean_img = read_img(clean_fn)
            noise_img = read_img(clean_fn.replace('GT_', 'NOISY_'))

        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        clean_img = self.transforms(clean_img)
        # clean_img = random_crop(clean_img)

        random.seed(seed) # apply this seed to img transforms
        torch.manual_seed(seed)
        noise_img = self.transforms(noise_img)
        # noise_img = random_crop(noise_img)
        return noise_img, clean_img


def load_image_pair_with_group(args):
    group_idx, img_path, numpy_flag = args
    noisy_path = img_path.replace('GT_', 'NOISY_')
    try:
        if numpy_flag:
            clean = read_img_np(img_path)
            noisy = read_img_np(noisy_path)
        else:
            clean = read_img(img_path)
            noisy = read_img(noisy_path)
        return (group_idx, (noisy, clean))
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None



class Syn_NTIRE(Dataset):
    def __init__(self, root_dir, sample_num, numpy_flag=False, patch_size=256):
        self.patch_size = patch_size
        self.numpy_flag = numpy_flag

        folders = sorted(glob.glob(root_dir + '/*'))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((patch_size, patch_size))
        ])
        total_clean_imgs = 0
        self.clean_fns = [[] for _ in range(sample_num)]
        for folder in folders:
            clean_imgs = sorted(glob.glob(folder + '/*GT_*'))
            total_clean_imgs += len(clean_imgs)
            print(f"{folder}: {len(clean_imgs)} GT images")
            for i, clean_img in enumerate(clean_imgs):
                self.clean_fns[i % sample_num].append(clean_img)
        print(f"Total GT images found: {total_clean_imgs}")

        self.grayscale = True
        # Preload images to memory
        self.data = []
        for img_list in self.clean_fns:
            img_path = random.choice(img_list)
            if numpy_flag:
                clean_img = read_img_np(img_path)
                noise_img = read_img_np(img_path.replace('GT_', 'NOISY_'))
            else:
                clean_img = read_img(img_path, self.grayscale)
                noise_img = read_img(img_path.replace('GT_', 'NOISY_'), self.grayscale)
            self.data.append((noise_img, clean_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noise_img, clean_img = self.data[idx]

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        clean_img = self.transforms(clean_img)

        random.seed(seed)
        torch.manual_seed(seed)
        noise_img = self.transforms(noise_img)

        return noise_img, clean_img


def _imread_uint8_CHW(path, mode="L"):
    img = Image.open(path)
    t = TF.pil_to_tensor(img)  # uint8, CxHxW, writable
    return t.contiguous()

class Syn_NTIRE_improved(Dataset):
    # def __init__(self, root_dir, sample_num, patch_size=256, rgb_mode="RGB", augment=True, draw_with_replacement=False):?
    def __init__(self, root_dir, sample_num, numpy_flag=False, patch_size=256, augment=True):
        self.patch_size = patch_size
        self.numpy_flag = numpy_flag
        self.grayscale = True

        folders = sorted(glob.glob(os.path.join(root_dir, '*')))
        all_gt_imgs = []
        for folder in folders:
            gt_imgs = sorted(glob.glob(os.path.join(folder, '*GT_*.png')))
            all_gt_imgs.extend(gt_imgs)

        total_imgs = len(all_gt_imgs)

        if sample_num > total_imgs:
            sample_num = total_imgs

        selected_gt_imgs = random.sample(all_gt_imgs, sample_num)

        self.data = []
        for img_path in selected_gt_imgs:
            # clean_img = read_img(img_path, self.grayscale)
            # noisy_img = read_img(img_path.replace('GT_', 'NOISY_'), self.grayscale)
            clean_img = _imread_uint8_CHW(img_path, self.grayscale)
            noisy_img = _imread_uint8_CHW(img_path.replace('GT_', 'NOISY_'), self.grayscale)
            self.data.append((noisy_img, clean_img))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((patch_size, patch_size))
        ])
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def _paired_random_crop_u8(self, a_u8, b_u8, ps):
        _, h, w = a_u8.shape
        if h < ps or w < ps:
            # Upscale minimally to fit patch (done in uint8 via nearest; you can switch to float+bilinear if you prefer)
            scale = max((ps + 0.5) / h, (ps + 0.5) / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            a_u8 = TF.interpolate(a_u8.unsqueeze(0).float(), size=(nh, nw), mode='bilinear', align_corners=False).to(torch.uint8).squeeze(0)
            b_u8 = TF.interpolate(b_u8.unsqueeze(0).float(), size=(nh, nw), mode='bilinear', align_corners=False).to(torch.uint8).squeeze(0)
            _, h, w = a_u8.shape
        i = torch.randint(0, h - ps + 1, (1,)).item()
        j = torch.randint(0, w - ps + 1, (1,)).item()
        return a_u8[:, i:i+ps, j:j+ps], b_u8[:, i:i+ps, j:j+ps]

    def _paired_aug_u8(self, a_u8, b_u8):
        if torch.rand(1).item() < 0.5:
            a_u8 = torch.flip(a_u8, dims=[2]); b_u8 = torch.flip(b_u8, dims=[2])  # horizontal
        if torch.rand(1).item() < 0.5:
            a_u8 = torch.flip(a_u8, dims=[1]); b_u8 = torch.flip(b_u8, dims=[1])  # vertical
        if torch.rand(1).item() < 0.5:
            a_u8 = a_u8.transpose(1,2); b_u8 = b_u8.transpose(1,2)                # transpose
        return a_u8.contiguous(), b_u8.contiguous()

    def __getitem__(self, idx):
        noisy_u8, clean_u8 = self.data[idx]

        noisy_u8, clean_u8 = self._paired_random_crop_u8(noisy_u8, clean_u8, self.patch_size)

        if self.augment:
            noisy_u8, clean_u8 = self._paired_aug_u8(noisy_u8, clean_u8)

        noisy = noisy_u8.to(torch.float32).div_(255.0).contiguous()
        clean = clean_u8.to(torch.float32).div_(255.0).contiguous()

        return noisy, clean

    # def __getitem__(self, idx):
    #     noisy_img, clean_img = self.data[idx]

    #     seed = random.randint(0, 2**31 - 1)
    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     clean_img = self.transforms(clean_img)

    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     noisy_img = self.transforms(noisy_img)

    #     return noisy_img, clean_img


class Syn_no_noisemap(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.ColorJitter(brightness=0.5, hue=0.35)])
        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def colour_transform(self, image, ground_truth, noise_map):
        im2tensor = transforms.ToTensor()
        image = im2tensor(image.copy())
        ground_truth = im2tensor(ground_truth.copy())
        noise_map = im2tensor(noise_map.copy())
        # test = TF.ColorJitter()
        # recolour = TF.ColorJitter(brightness=0.5, hue=0.35)

        recolour = transforms.ColorJitter(brightness=0.5, hue=0.35)
        transform = transforms.ColorJitter.get_params(
            recolour.brightness, recolour.contrast, recolour.saturation,
            recolour.hue
        )
        image = transform(image)
        ground_truth = transform(ground_truth)
        noise_map = transform(noise_map)
        return image, ground_truth, noise_map

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))

        if self.patch_size > 0:
            [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)
        return hwc_to_chw(noise_img), hwc_to_chw(clean_img)

class Syn(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        # sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB')) / 15.	# inverse scaling

        if self.patch_size > 0:
              [clean_img, noise_img] = get_patch([clean_img, noise_img], self.patch_size)

        noise_img = hwc_to_chw(noise_img)
        clean_img = hwc_to_chw(clean_img)
        # sigma_img = hwc_to_chw(sigma_img)
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.ones((3, self.patch_size, self.patch_size))
        return noise_img, clean_img, np.ones((3, self.patch_size, self.patch_size))

class test_data(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        # sigma_img = read_img(clean_fn.replace('GT_SRGB', 'SIGMA_SRGB')) / 15.	# inverse scaling

        if self.patch_size > 0:
                [clean_img, noise_img] = test_patch([clean_img, noise_img], 1500)

        noise_img = hwc_to_chw(noise_img)
        clean_img = hwc_to_chw(clean_img)
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.ones((3, self.patch_size, self.patch_size))
        return noise_img, clean_img, np.ones((3, self.patch_size, self.patch_size))

class benchmark_dataset(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=256, rgb=False, denoise_only=False):
        self.patch_size = patch_size
        self.denoise_only = denoise_only
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * len(folders)
        for i in range(len(folders)):
            self.clean_fns[i] = []
        if not rgb:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(1),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(patch_size)
                ])
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        if not denoise_only:
            for ind, folder in enumerate(folders):
                clean_imgs = glob.glob(folder + '/*GT_SRGB*')
                clean_imgs.sort()
                for clean_img in clean_imgs:
                    self.clean_fns[ind % len(folders)].append(clean_img)
        else:
            for ind, folder in enumerate(folders):
                clean_imgs = glob.glob(folder + '/*NOISY_SRGB*')
                clean_imgs.sort()
                for clean_img in clean_imgs:
                    self.clean_fns[ind % len(folders)].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        if not self.denoise_only:
            clean_fn = random.choice(self.clean_fns[idx])
            clean_img = read_img(clean_fn)
            noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
            clean_img = self.transforms(clean_img)
            noise_img = self.transforms(noise_img)
            # return noise_img, clean_img, np.ones((3, self.patch_size, self.patch_size))
            return noise_img, clean_img
        else:
            noisy_fn = self.clean_fns[idx][0]
            noisy_img = read_img(noisy_fn)
            noisy_img = self.transforms(noisy_img)
            return noisy_img, noisy_fn

class just_SIDD_clean(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):

        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)

        if self.patch_size > 0:
              [clean_img] = get_patch([clean_img], self.patch_size)

        clean_img = hwc_to_chw(clean_img)
        # return hwc_to_chw(noise_img), hwc_to_chw(clean_img), np.ones((3, self.patch_size, self.patch_size))
        return clean_img, np.ones((3, self.patch_size, self.patch_size))

# test_dataset = test_data_gpu('/home/bledc/Pictures/dataset/vol_1/', 50, patch_size=128)
class test_data_gpu(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((patch_size, patch_size)),
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            # clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs = glob.glob(folder + '/*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        # print(self.clean_fns[idx])
        # totensor = transforms.ToTensor()
        # crop = transforms.CenterCrop(128)
        # clean_fn = random.choice(self.clean_fns[idx])
        clean_fn = self.clean_fns[idx][0]

        clean_img = read_img(clean_fn)
        clean_img = self.transforms(clean_img)
        # clean_img = totensor(clean_img)
        # clean_img = crop(clean_img)
        noise_img = read_img(clean_fn.replace('GT_SRGB', 'NOISY_SRGB'))
        noise_img = self.transforms(noise_img)
        if self.noisemap:
            return noise_img, clean_img, np.zeros((3, noise_img.shape[1], noise_img.shape[1])), np.zeros((3, noise_img.shape[1], noise_img.shape[1]))
        else:
            return noise_img, clean_img

class test_data_NTIRE(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((patch_size, patch_size)),
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_*')
            clean_imgs.sort()

            for ind, clean_img in enumerate(clean_imgs):
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx][0]
        # for full dataset do random.choise(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        clean_img = self.transforms(clean_img)
        # clean_img = totensor(clean_img)
        # clean_img = crop(clean_img)
        noise_img = read_img(clean_fn.replace('GT_', 'NOISY_'))
        noise_img = self.transforms(noise_img)
        if self.noisemap:
            return noise_img, clean_img, np.zeros((3, noise_img.shape[1], noise_img.shape[1])), np.zeros((3, noise_img.shape[1], noise_img.shape[1]))
        else:
            return noise_img, clean_img


class train_video_gaussian(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((patch_size, patch_size)),
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_*')
            clean_imgs.sort()

            for ind, clean_img in enumerate(clean_imgs):
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx][0]
        # for full dataset do random.choise(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        clean_img = self.transforms(clean_img)
        # clean_img = totensor(clean_img)
        # clean_img = crop(clean_img)
        noise_img = read_img(clean_fn.replace('GT_', 'NOISY_'))
        noise_img = self.transforms(noise_img)
        if self.noisemap:
            return noise_img, clean_img, np.zeros((3, noise_img.shape[1], noise_img.shape[1])), np.zeros((3, noise_img.shape[1], noise_img.shape[1]))
        else:
            return noise_img, clean_img


class test_data_single_image(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop((patch_size, patch_size)),
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*.png*')
            clean_imgs.sort()

            for ind, clean_img in enumerate(clean_imgs):
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx][0]
        clean_img = read_img(clean_fn)
        clean_img = self.transforms(clean_img)
        return clean_img

class eval_NTIRE(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=500, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.CenterCrop(patch_size)
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*.png*')
            clean_imgs.sort()

            for ind, clean_img in enumerate(clean_imgs):
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx][0]
        # for full dataset do random.choise(self.clean_fns[idx])
        clean_img = read_img(clean_fn)
        clean_img = self.transforms(clean_img)
        return clean_img, os.path.basename(clean_fn)


class test_data_gpu2(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, noisemap=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * len(folders)
        for i in range(len(folders)):
            self.clean_fns[i] = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(patch_size),
            ])
        self.noisemap = noisemap
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*Noisy*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img) # EACH image takes a spot and lists all patches eg 420 images, 49 patches in each

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        noisy_fn = random.choice(self.clean_fns[idx])
        noisy_img = read_img(noisy_fn)
        head, tail = os.path.split(noisy_fn)
        clean_fn = glob.glob(head + '/*full*')
        clean_img = read_img(clean_fn[0])
        clean_img = self.transforms(clean_img)
        noisy_img = self.transforms(noisy_img)
        return noisy_img, clean_img

def noisy(image):
    row,col,ch = image.shape
    sigma = np.random.uniform(0.0, 50.0)
    sigma = sigma / 255
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

class just_gaussian(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, transform=None):
        self.patch_size = patch_size
        self.transform = transform
        folders = glob.glob(root_dir + '/*')
        folders.sort()

        self.clean_fns = [None] * sample_num
        for i in range(sample_num):
            self.clean_fns[i] = []

        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*GT_SRGB*')
            clean_imgs.sort()

            for clean_img in clean_imgs:
                self.clean_fns[ind % sample_num].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        clean_fn = random.choice(self.clean_fns[idx])
        clean_img = read_img(clean_fn)

        if self.patch_size > 0:
            [clean_img] = get_patch([clean_img], self.patch_size)
        noisy_img = noisy(clean_img)
        noisy_img = hwc_to_chw(noisy_img)
        noisy_img = np.clip(noisy_img, 0, 1)
        clean_img = hwc_to_chw(clean_img)
        # return noisy_img, clean_img, np.zeros((3, self.patch_size, self.patch_size)), np.zeros((3, self.patch_size, self.patch_size))
        return noisy_img, clean_img


class PairedDenoisingDataset(Dataset):
    def __init__(self, noisy_dir, noise_level, transform=None):
        self.noisy_dir = noisy_dir
        self.grayscale = True
        # self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "original")
        if self.grayscale:
            self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "grayscale_noise0")
            # self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "noise0_gray")
        else:
            self.clean_dir = os.path.join(os.path.abspath(os.path.join(noisy_dir, os.pardir)), "original")

        self.image_filenames = sorted([
            f for f in os.listdir(noisy_dir) if f.lower().endswith('.png')
        ])
        if not self.image_filenames:
            raise ValueError(f"No PNG images found in: {noisy_dir}")

        self.transform = transform or transforms.ToTensor()  # Default: convert to tensor
        self.noise_level = noise_level


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_img = read_img(noisy_path, self.grayscale)
        clean_img = read_img(clean_path, self.grayscale)
        # noisy_img = Image.open(noisy_path).convert('RGB')  # BSD68 is grayscale
        # clean_img = Image.open(clean_path).convert('RGB')

        return self.transform(noisy_img), self.transform(clean_img), self.noise_level
    
class PairedDenoisingDataset_eval(Dataset):
    def __init__(self, noisy_dir, noise_level, transform=None):
        self.noisy_dir = noisy_dir
        self.grayscale = True
        # self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "original")
        if self.grayscale:
            # self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "grayscale_noise0")
            self.clean_dir = os.path.join(os.path.dirname(noisy_dir), "noise0_gray")
        else:
            self.clean_dir = os.path.join(os.path.abspath(os.path.join(noisy_dir, os.pardir)), "original")

        self.image_filenames = sorted([
            f for f in os.listdir(noisy_dir) if f.lower().endswith('.png')
        ])
        if not self.image_filenames:
            raise ValueError(f"No PNG images found in: {noisy_dir}")

        self.transform = transform or transforms.ToTensor()  # Default: convert to tensor
        self.noise_level = noise_level


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_img = read_img(noisy_path, self.grayscale)
        clean_img = read_img(clean_path, self.grayscale)
        # noisy_img = Image.open(noisy_path).convert('RGB')  # BSD68 is grayscale
        # clean_img = Image.open(clean_path).convert('RGB')

        return self.transform(noisy_img), self.transform(clean_img), self.noise_level