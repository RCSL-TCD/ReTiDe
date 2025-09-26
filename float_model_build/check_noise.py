import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_noise_in_directory(directory_path):
    """
    Loads clean-noisy image pairs from a directory, calculates the residual,
    and prints the standard deviation of the noise.

    Args:
        directory_path (str): The path to the directory containing the images.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # Get a list of all files in the directory
    try:
        all_files = os.listdir(directory_path)
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return

    # Filter for noisy images to iterate through them
    noisy_files = sorted([f for f in all_files if f.startswith('NOISY_') and f.endswith('.png')])

    if not noisy_files:
        print("No noisy images (starting with 'NOISY_') found in the directory.")
        return

    print(f"Found {len(noisy_files)} noisy images. Analyzing pairs...")
    print("-" * 40)
    ave_std_dev = 0.0
    std_devs = []
    for noisy_filename in noisy_files:
        # Construct the corresponding ground truth filename
        base_name = noisy_filename[len('NOISY_'):]
        gt_filename = 'GT_' + base_name

        # Check if the corresponding ground truth image exists
        if gt_filename not in all_files:
            print(f"Warning: No matching GT image for '{noisy_filename}'")
            continue

        # Construct full file paths
        noisy_path = os.path.join(directory_path, noisy_filename)
        gt_path = os.path.join(directory_path, gt_filename)

        # Load images in grayscale
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Check if images were loaded successfully
        if noisy_img is None or gt_img is None:
            print(f"Error loading images for pair: {noisy_filename}, {gt_filename}")
            continue
        
        # Ensure images have the same dimensions
        if noisy_img.shape != gt_img.shape:
            print(f"Warning: Mismatched dimensions for '{base_name}'. Skipping.")
            continue

        # Convert images to float for accurate subtraction
        noisy_img_float = noisy_img.astype(np.float32)
        gt_img_float = gt_img.astype(np.float32)

        # Calculate the residual (noise)
        residual_img = noisy_img_float - gt_img_float

        # Measure the standard deviation of the residual
        noise_std_dev = np.std(residual_img)
        std_devs.append(noise_std_dev)
        print(f"Image: {base_name}, Noise Std Dev: {noise_std_dev:.4f}")
        ave_std_dev += noise_std_dev
    print(f"Average Noise Std Dev: {ave_std_dev / len(noisy_files):.4f}")
    
    if std_devs:
        plt.figure(figsize=(8, 5))
        plt.hist(std_devs, bins=20, color='blue', edgecolor='black')
        plt.title('Histogram of Noise Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        hist_path =  "noise_stddev_histogram.png"
        plt.savefig(hist_path)
        plt.close()

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Set the path to your directory containing the images here
    directory_to_analyze = "/data2/clement/images/datasets/denoising_challenge/DIV2K/DIV2k"
    # -----------------

    # Run the analysis
    analyze_noise_in_directory(directory_to_analyze)
