import torch

from models.my_models_0731 import UnetGenerator_hardware as UnetGenerator

model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8) #8color->6gray
weight_path = 'models/unet_f32_0814_color.pt'
if weight_path.endswith('.pt'):
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
torch.save(model.state_dict(), "unet_f32_0814_color_oldtorch.pt", _use_new_zipfile_serialization=False)
