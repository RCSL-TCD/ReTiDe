import torch

from models_0821 import UnetGenerator_3stage as UnetGenerator

model = UnetGenerator(1,1)
weight_path = 'unet_f32_0821_gray.pt'
if weight_path.endswith('.pt'):
    checkpoint = torch.load(weight_path, map_location='cpu')#, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
torch.save(model.state_dict(), "unet_f32_0821_gray_oldtorch.pt", _use_new_zipfile_serialization=False)

model = UnetGenerator(3,3)
weight_path = 'unet_f32_0821_color.pt'
if weight_path.endswith('.pt'):
    checkpoint = torch.load(weight_path, map_location='cpu')#, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
torch.save(model.state_dict(), "unet_f32_0821_color_oldtorch.pt", _use_new_zipfile_serialization=False)

