import torch
from ptflops import get_model_complexity_info

from models.models_0821 import UnetGenerator_3stage as Unet
unet_f32_weight_path = 'models/unet_f32_0821_color_oldtorch.pt' 

input_nc = 3
output_nc = 3
model = Unet(input_nc, output_nc)
model.load_state_dict(torch.load(unet_f32_weight_path))


input_size = (3, 256, 256)

flops, params = get_model_complexity_info(
    model, input_size, as_strings=True, print_per_layer_stat=True
)

print(f"FLOPs: {flops}")   # e.g., "4.1 GFLOPs"
print(f"Params: {params}") # e.g., "25.5 M"


import torch
from ptflops import get_model_complexity_info

from models.models_0812 import UnetGenerator_hardware as Unet
unet_f32_weight_path = 'models/unet_f32_0814_color_oldtorch.pt' 


model = Unet(input_nc=3, output_nc=3, num_downs=8)  
model.load_state_dict(torch.load(unet_f32_weight_path))


input_size = (3, 256, 256)

flops, params = get_model_complexity_info(
    model, input_size, as_strings=True, print_per_layer_stat=True
)

print(f"FLOPs: {flops}")   # e.g., "4.1 GFLOPs"
print(f"Params: {params}") # e.g., "25.5 M"