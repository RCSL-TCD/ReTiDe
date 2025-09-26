from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import re
import logging
from datetime import datetime
#from skimage.metrics import structural_similarity as ssim


import numpy as np
from scipy.ndimage import gaussian_filter

def calculate_ssim(img1, img2, data_range=255.0, window_size=11, sigma=1.5):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1, img2: Input images (same shape)
        data_range: The dynamic range of the images (255 for 8-bit images)
        window_size: Size of the Gaussian window
        sigma: Standard deviation for Gaussian window
    
    Returns:
        ssim_value: SSIM score
    """
    # Ensure images are float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants to avoid division by zero
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    gaussian_window = _create_gaussian_window(window_size, sigma)
    
    # Apply window to both images
    mu1 = _apply_window(img1, gaussian_window)
    mu2 = _apply_window(img2, gaussian_window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _apply_window(img1 ** 2, gaussian_window) - mu1_sq
    sigma2_sq = _apply_window(img2 ** 2, gaussian_window) - mu2_sq
    sigma12 = _apply_window(img1 * img2, gaussian_window) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def _create_gaussian_window(window_size, sigma):
    """Create a 2D Gaussian window."""
    x = np.arange(window_size) - (window_size - 1) / 2.0
    x = np.exp(-0.5 * (x ** 2) / (sigma ** 2))
    window = np.outer(x, x)
    return window / np.sum(window)

def _apply_window(image, window):
    """Apply 2D window to each channel of the image."""
    if len(image.shape) == 3:
        # For color images, apply window to each channel
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = _convolve2d(image[:, :, i], window)
        return result
    else:
        # For grayscale images
        return _convolve2d(image, window)

def _convolve2d(image, window):
    """2D convolution using FFT for efficiency."""
    from scipy.signal import fftconvolve
    return fftconvolve(image, window, mode='same')


# Configure logging
log_filename = f"denoise_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

_divider = '-------------------------------'

image_width = 256
image_height = 256
image_channels = 3  # Changed to 3 for RGB color

# Define noise levels
noise_levels = [5, 15, 25, 35, 45]  # Adjust according to your needs

def extract_image_index(filename):
    """Extract image index from filename"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def preprocess_fn(image_path, fix_scale):
    """Preprocessing function - handles color images"""
    # Read as color (BGR format)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning(f"Failed to read image at {image_path}")
        return None, None, None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if image.shape[:2] != (image_height, image_width):
        image = cv2.resize(image, (image_width, image_height))
    
    # Keep original image for PSNR calculation
    original_image = image.copy().astype(np.float32)
    
    # Preprocess for inference - scale to [0, fix_scale]
    processed_image = image * (1.0 / 255.0) * fix_scale
    processed_image = processed_image.astype(np.int8)
    
    return processed_image, original_image, os.path.basename(image_path)

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runDPU(id, start, dpu, img_list, out_q_list):
    """Run DPU inference - handles color I/O"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)   
    output_ndim = tuple(outputTensors[0].dims) 
    logger.info(f"Thread {id}: Input shape: {input_ndim}, Output shape: {output_ndim}")

    batchSize = input_ndim[0]
    n_of_images = len(img_list)
    count = 0
    write_index = start
    ids = []
    ids_max = 1000
    outputData = []
    
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    
    output_results = []

    while count < n_of_images:
        runSize = min(batchSize, n_of_images - count)

        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        for j in range(runSize):
            inputData[0][j, ...] = img_list[count + j][0].reshape(input_ndim[1:])

        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count += runSize

        if count < n_of_images and len(ids) < ids_max - 1:
            continue

        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]

            for j in range(ids[index][1]):
                output_img = outputData[index][0][j] 
                orig_index = write_index
                orig_filename = img_list[orig_index][2]
                output_results.append((output_img.copy(), orig_filename))
                out_q_list[write_index] = (output_img, orig_filename)
                write_index += 1

        ids = []

    return output_results

def app(noise_level, threads, model, folder):
    # Setup directories based on noise level
    base_dir = folder
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, f'noisy{noise_level}')
    should_denoised_dir = os.path.join(base_dir, f'should_denoise{noise_level}')
    denoised_dir = os.path.join(base_dir, f'denoise{noise_level}')
    
    # Create output directory
    if not os.path.exists(denoised_dir):
        os.makedirs(denoised_dir)
    
    logger.info(_divider)
    logger.info(f'Processing noise level: {noise_level}')
    logger.info(f'Noisy images directory: {noisy_dir}')
    logger.info(f'Clean images directory: {clean_dir}')
    logger.info(f'Float denoised directory: {should_denoised_dir}')
    logger.info(f'Output directory: {denoised_dir}')
    
    # Get sorted file list
    file_list = sorted(
        [f for f in os.listdir(noisy_dir) if f.endswith('.png')],
        key=lambda x: extract_image_index(x)
    )
    runTotal = len(file_list)
    logger.info(f"Found {runTotal} images in {noisy_dir}")

    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    output_fixpos = all_dpu_runners[0].get_output_tensors()[0].get_attr("fix_point")
    output_scale = 2**output_fixpos if output_fixpos is not None else 1.0

    logger.info(_divider)
    logger.info('Pre-processing %d images...', runTotal)
    img_list = []
    
    # Load clean images (as RGB)
    clean_images = {}
    for f in os.listdir(clean_dir):
        if f.endswith('.png'):
            idx = extract_image_index(f)
            if idx >= 0:
                clean_path = os.path.join(clean_dir, f)
                clean_image = cv2.imread(clean_path, cv2.IMREAD_COLOR)
                if clean_image is None:
                    logger.warning(f"Failed to read clean image at {clean_path}")
                    continue
                clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
                clean_image = cv2.resize(clean_image, (image_height, image_width))
                clean_images[idx] = clean_image.astype(np.float32)
    
    logger.info(f"Loaded {len(clean_images)} clean images")
    
    # Load float denoised images (as RGB)
    float_denoised_images = {}
    for f in os.listdir(should_denoised_dir):
        if f.endswith('.png'):
            idx = extract_image_index(f)
            if idx >= 0:
                float_path = os.path.join(should_denoised_dir, f)
                float_image = cv2.imread(float_path, cv2.IMREAD_COLOR)
                if float_image is None:
                    logger.warning(f"Failed to read float denoised image at {float_path}")
                    continue
                float_image = cv2.cvtColor(float_image, cv2.COLOR_BGR2RGB)
                float_image = cv2.resize(float_image, (image_width, image_height))
                float_denoised_images[idx] = float_image.astype(np.float32)
    
    logger.info(f"Loaded {len(float_denoised_images)} float denoised images")
    
    # Preprocess all input images
    skipped_images = 0
    for i, filename in enumerate(file_list):
        path = os.path.join(noisy_dir, filename)
        result = preprocess_fn(path, input_scale)
        if result[0] is None:
            skipped_images += 1
            continue
        preprocessed_img, orig_img, orig_filename = result
        img_list.append((preprocessed_img, orig_img, orig_filename))
    
    if skipped_images > 0:
        logger.warning("Skipped %d images due to read errors", skipped_images)

    logger.info(_divider)
    logger.info('Starting %d threads...', threads)
    threadAll = []
    start = 0
    for i in range(threads):
        if i == threads - 1:
            end = len(img_list)
        else:
            end = start + (len(img_list) // threads)
        in_q = img_list[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q, out_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    logger.info(_divider)
    logger.info("Throughput=%.2f fps, total frames = %d, time=%.4f seconds", fps, runTotal, timetotal)

    # Save denoised images
    if not os.path.exists(denoised_dir):
        os.makedirs(denoised_dir)

    # Initialize metrics
    average_psnr = 0
    average_mse = 0
    average_ssim = 0
    
    average_noisy_psnr = 0
    average_noisy_mse = 0
    average_noisy_ssim = 0
    
    average_float_psnr = 0
    average_float_mse = 0
    average_float_ssim = 0

    processed_count = 0
    skipped_count = 0

    for i in range(len(out_q)):
        if out_q[i] is None:
            logger.warning("No output for index %d", i)
            skipped_count += 1
            continue
            
        output_img, filename = out_q[i]
        idx = extract_image_index(filename)
        
        if idx not in clean_images:
            logger.warning("No clean image for index %d (from %s)", idx, filename)
            skipped_count += 1
            continue

        # Convert denoised output to float in [0,255] range
        denoised_float = (output_img.astype(np.float32) / output_scale) * 255.0
        denoised_float = np.clip(denoised_float, 0, 255)
        
        # Get original noisy image
        noisy_float = img_list[i][1]
        
        # Get clean reference
        clean_ref = clean_images[idx]
        
        # Get float denoised result
        if idx not in float_denoised_images:
            logger.warning("Float denoised image not found for index %d", idx)
            skipped_count += 1
            continue
        float_denoised = float_denoised_images[idx]
        
        # Verify shapes
        if denoised_float.shape != clean_ref.shape:
            logger.warning(f"Shape mismatch at index {idx}: denoised {denoised_float.shape} vs clean {clean_ref.shape}")
            skipped_count += 1
            continue
        
        # Calculate metrics
        # 1. Noisy image quality
        noisy_mse = np.mean((clean_ref - noisy_float) ** 2)
        noisy_psnr = 10 * np.log10((255.0 ** 2) / noisy_mse) if noisy_mse > 1e-10 else 100
        noisy_ssim = calculate_ssim(clean_ref, noisy_float)
        #def calculate_ssim(img1, img2, data_range=255.0, window_size=11, sigma=1.5):
     
        # 2. Quantized model quality
        mse = np.mean((clean_ref - denoised_float) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 1e-10 else 100
        ssim_val = calculate_ssim(clean_ref, denoised_float)

        # 3. Float model quality
        float_mse = np.mean((clean_ref - float_denoised) ** 2)
        float_psnr = 10 * np.log10((255.0 ** 2) / float_mse) if float_mse > 1e-10 else 100
        float_ssim = calculate_ssim(clean_ref, float_denoised)
        
        # Save denoised image (convert back to BGR for OpenCV)
        denoised_bgr = cv2.cvtColor(denoised_float.astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(denoised_dir, f'denoised_{idx}.png')
        cv2.imwrite(output_path, denoised_bgr)
        
        # Accumulate metrics
        average_psnr += psnr
        average_mse += mse
        average_ssim += ssim_val
        
        average_noisy_psnr += noisy_psnr
        average_noisy_mse += noisy_mse
        average_noisy_ssim += noisy_ssim
        
        average_float_psnr += float_psnr
        average_float_mse += float_mse
        average_float_ssim += float_ssim
        
        processed_count += 1

    if processed_count == 0:
        logger.error("No valid images processed for metrics calculation")
        return None

    # Calculate averages
    average_psnr /= processed_count
    average_mse /= processed_count
    average_ssim /= processed_count
    
    average_noisy_psnr /= processed_count
    average_noisy_mse /= processed_count
    average_noisy_ssim /= processed_count
    
    average_float_psnr /= processed_count
    average_float_mse /= processed_count
    average_float_ssim /= processed_count

    # Calculate improvements
    average_psnr_improvement = average_psnr - average_noisy_psnr
    average_mse_improvement = average_noisy_mse - average_mse
    average_ssim_improvement = average_ssim - average_noisy_ssim
    
    average_float_psnr_improvement = average_float_psnr - average_noisy_psnr
    average_float_mse_improvement = average_noisy_mse - average_float_mse
    average_float_ssim_improvement = average_float_ssim - average_noisy_ssim

    # Log results
    logger.info('Denoised images saved to: %s', denoised_dir)
    logger.info('Processed %d images, skipped %d', processed_count, skipped_count)
    logger.info('--- Noise Level: %d ---', noise_level)
    logger.info('Average Noisy PSNR: %.2f dB', average_noisy_psnr)
    logger.info('Average Noisy MSE: %.4f', average_noisy_mse)
    logger.info('Average Noisy SSIM: %.4f', average_noisy_ssim)
    logger.info('---')
    logger.info('Average Float32 Denoised PSNR: %.2f dB', average_float_psnr)
    logger.info('Average Float32 Denoised MSE: %.4f', average_float_mse)
    logger.info('Average Float32 Denoised SSIM: %.4f', average_float_ssim)
    logger.info('Float32 PSNR Improvement: %.2f dB', average_float_psnr_improvement)
    logger.info('Float32 MSE Improvement: %.4f', average_float_mse_improvement)
    logger.info('Float32 SSIM Improvement: %.4f', average_float_ssim_improvement)
    logger.info('---')
    logger.info('Average Quantized Denoised PSNR: %.2f dB', average_psnr)
    logger.info('Average Quantized Denoised MSE: %.4f', average_mse)
    logger.info('Average Quantized Denoised SSIM: %.4f', average_ssim)
    logger.info('Quantized PSNR Improvement: %.2f dB', average_psnr_improvement)
    logger.info('Quantized MSE Improvement: %.4f', average_mse_improvement)
    logger.info('Quantized SSIM Improvement: %.4f', average_ssim_improvement)
    logger.info('---')
    logger.info('PSNR Gap (Float32 - Quantized): %.2f dB', (average_float_psnr - average_psnr))
    logger.info('MSE Gap (Quantized - Float32): %.4f', (average_mse - average_float_mse))
    logger.info('SSIM Gap (Float32 - Quantized): %.4f', (average_float_ssim - average_ssim))
    logger.info('Done.')
    logger.info(_divider)
    
    # Return results for summary
    return {
        'noise_level': noise_level,
        'processed_count': processed_count,
        'average_noisy_psnr': average_noisy_psnr,
        'average_noisy_ssim': average_noisy_ssim,
        'average_float_psnr': average_float_psnr,
        'average_float_ssim': average_float_ssim,
        'average_quantized_psnr': average_psnr,
        'average_quantized_ssim': average_ssim,
        'float_psnr_improvement': average_float_psnr_improvement,
        'float_ssim_improvement': average_float_ssim_improvement,
        'quantized_psnr_improvement': average_psnr_improvement,
        'quantized_ssim_improvement': average_ssim_improvement,
        'psnr_gap': average_float_psnr - average_psnr,
        'ssim_gap': average_float_ssim - average_ssim
    }

def main():
    ap = argparse.ArgumentParser()  
    ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='UnetGenerator_u50.xmodel', help='Path of xmodel. Default is CNN_zcu102.xmodel')
    ap.add_argument('-f', '--folder',    type=str, default='test_set', help='Dataset path. Default is test_set')
    args = ap.parse_args()  
    
    logger.info('Command line options:')
    logger.info(' --threads   : %d', args.threads)
    logger.info(' --model     : %s', args.model)
    logger.info(' --folder    : %s', args.folder)
    
    # Store results for all noise levels
    all_results = []
    
    # Process all noise levels
    for noise_level in noise_levels:
        result = app(noise_level, args.threads, args.model, args.folder)
        if result:
            all_results.append(result)
    
    # Print summary
    logger.info('\n\n' + _divider)
    logger.info('SUMMARY OF RESULTS FOR ALL NOISE LEVELS')
    logger.info(_divider)
    
    # PSNR Summary Table
    logger.info('PSNR RESULTS:')
    logger.info('Noise Level | Processed | Noisy PSNR | Float PSNR | Quant PSNR | Float Imp | Quant Imp | PSNR Gap')
    logger.info('------------------------------------------------------------------------------------------------')
    for res in all_results:
        logger.info(f"{res['noise_level']:^11} | {res['processed_count']:^9} | "
                    f"{res['average_noisy_psnr']:^10.2f} | {res['average_float_psnr']:^9.2f} | "
                    f"{res['average_quantized_psnr']:^10.2f} | {res['float_psnr_improvement']:^8.2f} | "
                    f"{res['quantized_psnr_improvement']:^9.2f} | {res['psnr_gap']:^8.2f}")
    
    logger.info('\n' + _divider)
    
    # SSIM Summary Table
    logger.info('SSIM RESULTS:')
    logger.info('Noise Level | Processed | Noisy SSIM | Float SSIM | Quant SSIM | Float Imp | Quant Imp | SSIM Gap')
    logger.info('------------------------------------------------------------------------------------------------')
    for res in all_results:
        logger.info(f"{res['noise_level']:^11} | {res['processed_count']:^9} | "
                    f"{res['average_noisy_ssim']:^10.4f} | {res['average_float_ssim']:^9.4f} | "
                    f"{res['average_quantized_ssim']:^10.4f} | {res['float_ssim_improvement']:^8.4f} | "
                    f"{res['quantized_ssim_improvement']:^9.4f} | {res['ssim_gap']:^8.4f}")
    
    logger.info(_divider)
    logger.info('Log saved to: %s', log_filename)

if __name__ == '__main__':
    main()