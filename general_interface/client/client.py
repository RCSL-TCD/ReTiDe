import requests
import json
import os
import sys
from datetime import datetime
import shutil

from configs.server_config import server_ip, port
from utils.tasks import multiple_f32_inference, test_api, single_f32_inference, video_denoise_inference
from utils.tools import clean_results


if __name__ == "__main__":
    test_api()
    clean_results()
    image_path = "test_img/BSD100C/noisy35/noisy_0.png"
    single_f32_inference(image_path)
 #   folder_path = "test_img/BSD100C/noisy35"
#    multiple_f32_inference(folder_path)
    folder_path = "test_img/clips"
    multiple_f32_inference(folder_path)
    # video_path = "test_img/video/output/cropped_noisy.mp4"
    # video_denoise_inference(video_path)