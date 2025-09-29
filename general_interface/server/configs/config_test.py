import os

UPLOAD_FOLDER = 'uploads/images'
PROCESSED_FOLDER = 'processed/images'
BATCH_SIZE = 32
INPUT_NC = 3
OUTPUT_NC = 3
NUM_DOWNS = 8
UNET_F32_WEIGHT_PATH = 'models/Color/unet_f32_0814_color_oldtorch.pt'

def init_configs():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)


