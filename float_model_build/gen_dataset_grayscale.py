import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random

def add_gaussian_noise(image, std_dev):
    """
    Adds Gaussian noise to a grayscale PIL image.
    """
    image = image.convert('L')  # Ensure grayscale
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std_dev, img_array.shape)
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255)
    return Image.fromarray(noisy_array.astype('uint8'), mode='L')

def process_single_image(args):
    filename, input_dir, output_dir = args

    input_path = os.path.join(input_dir, filename)
    clean_output_path = os.path.join(output_dir, filename)
    noisy_output_path = os.path.join(output_dir, filename.replace("GT_", "NOISY_", 1))

    try:
        with Image.open(input_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img.save(clean_output_path)

            sigma = random.uniform(0, 50)  # Random std dev
            noisy_img = add_gaussian_noise(img, sigma)
            noisy_img.save(noisy_output_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def process_images_parallel(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found at '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png') and f.startswith('GT_')]
    if not image_files:
        print(f"No 'GT_*.png' images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images to process (σ ∈ [0, 50])")

    task_args = [(filename, input_dir, output_dir) for filename in image_files]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_image, task_args), total=len(task_args), desc="Generating noisy images"))

def main():
    parser = argparse.ArgumentParser(description="Create a grayscale denoising dataset with random Gaussian noise.")
    parser.add_argument("input_dir", type=str, help="Directory containing clean PNG images with names like 'GT_*.png'.")
    parser.add_argument("output_dir", type=str, help="Directory to store both clean and noisy images.")
    args = parser.parse_args()

    process_images_parallel(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
