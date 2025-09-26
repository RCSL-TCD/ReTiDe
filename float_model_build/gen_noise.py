import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def add_gaussian_noise(image, std_dev):
    """
    Adds Gaussian noise to a PIL image.

    Args:
        image (PIL.Image.Image): The input image.
        std_dev (float): The standard deviation of the Gaussian noise.

    Returns:
        PIL.Image.Image: The noisy image.
    """
    if image.mode == 'L':
        # Grayscale image
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, std_dev, img_array.shape)
        noisy_array = img_array + noise
    elif image.mode == 'RGB':
        # Color image
        img_array = np.array(image, dtype=np.float32)
        noise = np.random.normal(0, std_dev, img_array.shape)
        noisy_array = img_array + noise
    else:
        # For other modes, convert to RGB first
        rgb_image = image.convert('RGB')
        img_array = np.array(rgb_image, dtype=np.float32)
        noise = np.random.normal(0, std_dev, img_array.shape)
        noisy_array = img_array + noise

    # Clip values to the valid range [0, 255] and convert back to uint8
    noisy_array = np.clip(noisy_array, 0, 255)
    noisy_image = Image.fromarray(noisy_array.astype('uint8'), image.mode if image.mode in ['L', 'RGB'] else 'RGB')
    
    return noisy_image

def process_images(directory):
    """
    Processes all 'GT_*.png' images in a directory, adds random Gaussian noise,
    and saves them as 'NOISY_*.png'.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    image_files = [f for f in os.listdir(directory) if f.startswith('GT_') and f.endswith('.png')]

    if not image_files:
        print(f"No 'GT_*.png' images found in '{directory}'")
        return

    print(f"Found {len(image_files)} images to process.")

    for filename in tqdm(image_files, desc="Processing images"):
        try:
            # Define paths
            input_path = os.path.join(directory, filename)
            output_filename = filename.replace('GT_', 'NOISY_', 1)
            output_path = os.path.join(directory, output_filename)

            # Open image
            with Image.open(input_path) as img:
                # Choose a random standard deviation
                sigma = random.uniform(0, 50)

                # Add noise
                noisy_img = add_gaussian_noise(img, sigma)

                # Save the new image
                noisy_img.save(output_path)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def process_single_image(args):
    filename, directory = args
    input_path = os.path.join(directory, filename)
    output_filename = filename.replace('GT_', 'NOISY_', 1)
    output_path = os.path.join(directory, output_filename)

    try:
        with Image.open(input_path) as img:
            sigma = random.uniform(0, 50)
            noisy_img = add_gaussian_noise(img, sigma)
            noisy_img.save(output_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def process_images_parallel(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    image_files = [f for f in os.listdir(directory) if f.startswith('GT_') and f.endswith('.png')]

    if not image_files:
        print(f"No 'GT_*.png' images found in '{directory}'")
        return

    print(f"Found {len(image_files)} images to process.")

    # Prepare args as (filename, directory) tuples
    task_args = [(filename, directory) for filename in image_files]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_image, task_args), total=len(task_args), desc="Processing images"))


def main():
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to PNG images in a directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory containing 'GT_*.png' images."
    )
    args = parser.parse_args()
    # process_images(args.directory)
    process_images_parallel(args.directory)

if __name__ == "__main__":
    main()