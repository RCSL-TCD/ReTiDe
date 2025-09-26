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
image_channels = 1  # Grayscale

# Define noise levels
noise_levels = [15, 25, 35, 50]  # Adjust according to your needs

def extract_image_index(filename):
    """Extract image index from filename"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def preprocess_fn(image_path, fix_scale):
    """Preprocessing function - handles grayscale images"""
    # Read as grayscale directly
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning(f"Failed to read image at {image_path}")
        return None, None, None
    
    # Resize if needed
    if image.shape != (image_height, image_width):
        image = cv2.resize(image, (image_width, image_height))
    
    # Add channel dimension for processing
    image = np.expand_dims(image, axis=-1)
    
    # Keep original image for PSNR calculation
    original_image = image.copy().astype(np.float32)
    
    # Preprocess for inference
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
    """Run DPU inference - handles grayscale I/O"""
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
    
    # Load clean images
    clean_images = {}
    for f in os.listdir(clean_dir):
        if f.endswith('.png'):
            idx = extract_image_index(f)
            if idx >= 0:
                clean_path = os.path.join(clean_dir, f)
                clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                if clean_image is None:
                    logger.warning(f"Failed to read clean image at {clean_path}")
                    continue
                clean_image = cv2.resize(clean_image, (image_height, image_width))
                clean_image = np.expand_dims(clean_image, axis=-1)  # Add channel dimension
                clean_images[idx] = clean_image.astype(np.float32)
    
    logger.info(f"Loaded {len(clean_images)} clean images")
    
    # Load float denoised images
    float_denoised_images = {}
    for f in os.listdir(should_denoised_dir):
        if f.endswith('.png'):
            idx = extract_image_index(f)
            if idx >= 0:
                float_path = os.path.join(should_denoised_dir, f)
                float_image = cv2.imread(float_path, cv2.IMREAD_GRAYSCALE)
                if float_image is None:
                    logger.warning(f"Failed to read float denoised image at {float_path}")
                    continue
                float_image = cv2.resize(float_image, (image_width, image_height))
                float_image = np.expand_dims(float_image, axis=-1)  # Add channel dimension
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

    average_psnr = 0
    average_mse = 0
    average_noisy_psnr = 0
    average_noisy_mse = 0
    average_float_psnr = 0
    average_float_mse = 0

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

        # ====== 关键修正部分：保持浮点精度计算PSNR ======
        # 直接转换到浮点范围[0,255]，不转uint8
        denoised_float = (output_img.astype(np.float32) / output_scale) * 255.0
        denoised_float = np.clip(denoised_float, 0, 255)
        
        # 获取预处理时保存的原始噪声图（浮点格式）
        noisy_float = img_list[i][1]
        
        # 获取干净参考图
        clean_ref = clean_images[idx]
        
        # 获取浮点模型去噪结果
        if idx not in float_denoised_images:
            logger.warning("Float denoised image not found for index %d", idx)
            skipped_count += 1
            continue
        float_denoised = float_denoised_images[idx]
        
        # 确保所有图像形状一致
        if denoised_float.shape != clean_ref.shape:
            logger.warning(f"Shape mismatch at index {idx}: denoised {denoised_float.shape} vs clean {clean_ref.shape}")
            skipped_count += 1
            continue
        
        # ====== 计算指标 ======
        # 1. 噪声图像质量
        noisy_mse = np.mean((clean_ref - noisy_float) ** 2)
        noisy_psnr = 10 * np.log10((255.0 ** 2) / noisy_mse) if noisy_mse > 1e-10 else 100
        
        # 2. 量化模型质量
        mse = np.mean((clean_ref - denoised_float) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 1e-10 else 100
        
        # 3. 浮点模型质量
        float_mse = np.mean((clean_ref - float_denoised) ** 2)
        float_psnr = 10 * np.log10((255.0 ** 2) / float_mse) if float_mse > 1e-10 else 100
        
        # 保存图像（仅保存时转uint8）
        denoised_uint8 = denoised_float.astype(np.uint8)
        if denoised_uint8.shape[-1] == 1:
            denoised_uint8 = denoised_uint8.squeeze(axis=-1)
        output_path = os.path.join(denoised_dir, f'denoised_{idx}.png')
        cv2.imwrite(output_path, denoised_uint8)
        # ====== 修正结束 ======
        
        # 累加指标
        average_psnr += psnr
        average_mse += mse
        average_noisy_psnr += noisy_psnr
        average_noisy_mse += noisy_mse
        average_float_psnr += float_psnr
        average_float_mse += float_mse
        
        processed_count += 1

    if processed_count == 0:
        logger.error("No valid images processed for PSNR calculation")
        return None

    # Calculate averages
    average_psnr /= processed_count
    average_mse /= processed_count
    average_noisy_psnr /= processed_count
    average_noisy_mse /= processed_count
    average_float_psnr /= processed_count
    average_float_mse /= processed_count

    # Calculate improvements
    average_psnr_improvement = average_psnr - average_noisy_psnr
    average_mse_improvement = average_noisy_mse - average_mse
    average_float_psnr_improvement = average_float_psnr - average_noisy_psnr
    average_float_mse_improvement = average_noisy_mse - average_float_mse

    # Log results
    logger.info('Denoised images saved to: %s', denoised_dir)
    logger.info('Processed %d images, skipped %d', processed_count, skipped_count)
    logger.info('--- Noise Level: %d ---', noise_level)
    logger.info('Average Noisy PSNR: %.2f dB', average_noisy_psnr)
    logger.info('Average Noisy MSE: %.4f', average_noisy_mse)
    logger.info('---')
    logger.info('Average Float32 Denoised PSNR: %.2f dB', average_float_psnr)
    logger.info('Average Float32 Denoised MSE: %.4f', average_float_mse)
    logger.info('Float32 PSNR Improvement: %.2f dB', average_float_psnr_improvement)
    logger.info('Float32 MSE Improvement: %.4f', average_float_mse_improvement)
    logger.info('---')
    logger.info('Average Quantized Denoised PSNR: %.2f dB', average_psnr)
    logger.info('Average Quantized Denoised MSE: %.4f', average_mse)
    logger.info('Quantized PSNR Improvement: %.2f dB', average_psnr_improvement)
    logger.info('Quantized MSE Improvement: %.4f', average_mse_improvement)
    logger.info('---')
    logger.info('PSNR Gap (Float32 - Quantized): %.2f dB', (average_float_psnr - average_psnr))
    logger.info('MSE Gap (Quantized - Float32): %.4f', (average_mse - average_float_mse))
    logger.info('Done.')
    logger.info(_divider)
    
    # Return results for summary
    return {
        'noise_level': noise_level,
        'processed_count': processed_count,
        'average_noisy_psnr': average_noisy_psnr,
        'average_float_psnr': average_float_psnr,
        'average_quantized_psnr': average_psnr,
        'float_improvement': average_float_psnr_improvement,
        'quantized_improvement': average_psnr_improvement,
        'psnr_gap': average_float_psnr - average_psnr
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
    logger.info(' --folder    " %s', args.folder)
    
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
    logger.info('Noise Level | Processed | Noisy PSNR | Float PSNR | Quant PSNR | Float Imp | Quant Imp | PSNR Gap')
    logger.info('------------------------------------------------------------------------------------------------')
    for res in all_results:
        logger.info(f"{res['noise_level']:^11} | {res['processed_count']:^9} | "
                    f"{res['average_noisy_psnr']:^10.2f} | {res['average_float_psnr']:^9.2f} | "
                    f"{res['average_quantized_psnr']:^10.2f} | {res['float_improvement']:^8.2f} | "
                    f"{res['quantized_improvement']:^9.2f} | {res['psnr_gap']:^8.2f}")
    logger.info(_divider)
    logger.info('Log saved to: %s', log_filename)

if __name__ == '__main__':
    main()