import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def add_gaussian_noise(image, std_dev):
    gray_img = image.convert('L')
    img_array = np.array(gray_img, dtype=np.float32)
    noise = np.random.normal(0, std_dev, img_array.shape)
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255)
    return Image.fromarray(noisy_array.astype('uint8'), mode='L')


def process_single_image(args):
    filename, input_dir, output_dir, sigma = args
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)  # same filename

    try:
        with Image.open(input_path) as img:
            noisy_img = add_gaussian_noise(img, sigma)
            noisy_img.save(output_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")


def process_images_parallel(input_dir, sigma):
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found at '{input_dir}'")
        return

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
    if not image_files:
        print(f"No PNG images found in '{input_dir}'")
        return

    parent_dir = os.path.dirname(input_dir.rstrip("/"))
    output_dir = os.path.join(parent_dir, f"grayscale_noise{int(sigma)}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(image_files)} images to process.")
    print(f"Saving grayscale noisy images to: {output_dir}")

    task_args = [(filename, input_dir, output_dir, sigma) for filename in image_files]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_image, task_args), total=len(task_args), desc="Generating grayscale noisy images"))


def main():
    parser = argparse.ArgumentParser(description="Convert images to grayscale and add Gaussian noise.")
    parser.add_argument("directory", type=str, help="Path to the input directory containing PNG images.")
    parser.add_argument("--sigma", type=float, required=True, help="Standard deviation of Gaussian noise.")
    args = parser.parse_args()

    process_images_parallel(args.directory, args.sigma)


if __name__ == "__main__":
    main()
