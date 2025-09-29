from flask import Blueprint, jsonify, request, send_from_directory, current_app
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.models_0812 import UnetGenerator_hardware as Unet
from utils.tools import clean_results, allowed_file
from configs.config_test import *

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import sys

import cv2
import numpy as np
import os
import uuid
import shutil
from werkzeug.utils import secure_filename
from flask import request, jsonify, send_file
import tempfile
import concurrent.futures
api_bp = Blueprint('api', __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'running with {device}')

model = Unet(input_nc=INPUT_NC, output_nc=OUTPUT_NC, num_downs=NUM_DOWNS) 
model.load_state_dict(torch.load(UNET_F32_WEIGHT_PATH, map_location=device))
model.to(device)
print('model loaded')
model.eval()

def process_image(image_path, output_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))



        transform = transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(image_tensor)

        output_tensor = output_tensor.clamp(0, 1).cpu().squeeze(0)
        output_image = transforms.ToPILImage()(output_tensor)
        output_image.save(output_path, format="PNG")

        return True
    except Exception as e:
        print(f"process images error: {e}")
        return False

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

def fpga_process_image(image_path, output_path, xmodel_path):
    try:
        # load DPU model
        g = xir.Graph.deserialize(xmodel_path)
        subgraphs = get_child_subgraph_dpu(g)
        dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

        # get input and output tensor information
        inputTensors = dpu_runner.get_input_tensors()
        outputTensors = dpu_runner.get_output_tensors()
        
        input_ndim = tuple(inputTensors[0].dims)
        output_ndim = tuple(outputTensors[0].dims)
        
        # get fix point and scale
        input_fixpos = inputTensors[0].get_attr("fix_point")
        input_scale = 2**input_fixpos
        
        output_fixpos = outputTensors[0].get_attr("fix_point")
        output_scale = 2**output_fixpos if output_fixpos is not None else 1.0
        
        print(f"Input shape: {input_ndim}, Output shape: {output_ndim}")
        print(f"Input scale: {input_scale}, Output scale: {output_scale}")
        
        # load and preprocess image
        # read as BGR format
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to read image at {image_path}")
            return False
        
        # convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # check and resize image to model input size
        input_height, input_width = input_ndim[1:3]
        if image_rgb.shape[:2] != (input_height, input_width):
            image_resized = cv2.resize(image_rgb, (input_width, input_height))
        else:
            image_resized = image_rgb

        # preprocess: normalize and quantize
        processed_image = image_resized.astype(np.float32) * (1.0 / 255.0) * input_scale
        processed_image = processed_image.astype(np.int8)

        # prepare input data (batch_size=1)
        input_data = [np.empty(input_ndim, dtype=np.int8, order="C")]
        input_data[0][0, ...] = processed_image.reshape(input_ndim[1:])
        
        # prepare output data
        output_data = [np.empty(output_ndim, dtype=np.int8, order="C")]
        
        # inference
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        
        # post
        output_img = output_data[0][0]  # get the first (and only) batch item
        
        # dequantize and convert to uint8
        denoised_float = (output_img.astype(np.float32) / output_scale) * 255.0
        denoised_float = np.clip(denoised_float, 0, 255).astype(np.uint8)
        
        # convert RGB to BGR for saving
        denoised_bgr = cv2.cvtColor(denoised_float, cv2.COLOR_RGB2BGR)

        # ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save image
        cv2.imwrite(output_path, denoised_bgr)
        
        print(f"Successfully processed image: {image_path} -> {output_path}")
        
        # clean up
        del dpu_runner
        
        return True
        
    except Exception as e:
        print(f"FPGA process image error: {e}")
        return False

@api_bp.route('/hello', methods=['GET'])
def hello():
    return jsonify(message="Interface test ok!")

@api_bp.route('/f32_inference', methods=['POST'])
def f32_inference():
    clean_results()
    try:
        if 'image' not in request.files:
            return jsonify(error="No image file provided"), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify(error="No selected file"), 400
        
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{timestamp}_{unique_id}{ext}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            processed_filename = f"processed_{name}_{timestamp}_{unique_id}.png"
            processed_filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename)
            
            process_success = process_image(filepath, processed_filepath)
            
            result = {
                "status": "success",
                "message": "Image uploaded and processed successfully",
                "original_filename": original_filename,
                "saved_filename": filename,
                "filepath": filepath,
                "inference_result": {
                    "class": "processed_image",
                    "confidence": 0.95
                }
            }
            
            if process_success:
                result["processed_image_url"] = f"http://{request.host}/api/processed/{processed_filename}"
            
            return jsonify(result)
        else:
            return jsonify(error="File type not allowed"), 400
        
    except Exception as e:
        return jsonify(error=f"Failed to process image: {str(e)}"), 500

@api_bp.route('/f32_inference_multiple', methods=['POST'])
def f32_inference_multiple():
    clean_results()
    try:
        if 'images' not in request.files:
            return jsonify(error="No image files provided"), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify(error="No selected files"), 400
        
        results = []
        valid_files = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                valid_files += 1
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                name, ext = os.path.splitext(original_filename)
                filename = f"{name}_{timestamp}_{unique_id}{ext}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                
                file.save(filepath)
                
                processed_filename = f"processed_{name}_{timestamp}_{unique_id}.png"
                processed_filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename)
                
                process_success = process_image(filepath, processed_filepath)
                
                file_result = {
                    "original_filename": original_filename,
                    "saved_filename": filename,
                    "filepath": filepath,
                    "inference_result": {
                        "class": "processed_image",
                        "confidence": 0.95
                    }
                }
                
                if process_success:
                    file_result["processed_image_url"] = f"http://{request.host}/api/processed/{processed_filename}"
                
                results.append(file_result)
        
        if valid_files == 0:
            return jsonify(error="No valid image files found"), 400
        
        response_data = {
            "status": "success",
            "message": f"Processed {valid_files} images successfully",
            "total_images": valid_files,
            "results": results
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify(error=f"Failed to process images: {str(e)}"), 500

@api_bp.route('/processed/<filename>')
def get_processed_image(filename):
    try:
        return send_from_directory(current_app.config['PROCESSED_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify(error="Processed image not found"), 404

@api_bp.route('/uploads/<filename>')
def get_uploaded_image(filename):
    try:
        return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify(error="Uploaded image not found"), 404
    
@api_bp.route('/video_denoise', methods=['POST'])
def video_denoise():
    """
    处理MP4视频：分割为256x256图片片段 -> 降噪处理 -> 重新拼接为视频
    """
    try:
        # 检查文件
        if 'video' not in request.files:
            return jsonify(error="No video file provided"), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify(error="No selected file"), 400
        
        # 检查文件类型
        if not file.filename.lower().endswith('.mp4'):
            return jsonify(error="Only MP4 files are supported"), 400
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        original_video_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(original_video_path)
        
        print(f"✓ Video saved to: {original_video_path}")
        
        # 处理视频
        result = process_video_pipeline(original_video_path, temp_dir)
        
        if result['success']:
            # 返回处理后的视频
            response = send_file(
                result['processed_video_path'],
                as_attachment=True,
                download_name=f"denoised_{secure_filename(file.filename)}",
                mimetype='video/mp4'
            )
            
            # 清理临时文件（在实际应用中可能需要延迟清理）
            # shutil.rmtree(temp_dir, ignore_errors=True)
            
            return response
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify(error=result['error']), 500
            
    except Exception as e:
        # 确保清理临时文件
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(error=f"Failed to process video: {str(e)}"), 500

def process_video_pipeline(video_path, temp_dir):
    """
    视频处理流水线：分割 -> 降噪 -> 拼接
    """
    try:
        # 步骤1: 分割视频为256x256图片片段
        print("Step 1: Splitting video into 256x256 frames...")
        split_result = split_video_to_frames(video_path, temp_dir)
        if not split_result['success']:
            return split_result
        
        frames_info = split_result['frames_info']
        frames_dir = split_result['frames_dir']
        
        # 步骤2: 对每个图片片段进行降噪处理
        print("Step 2: Denoising frames...")
        denoise_result = denoise_frames(frames_info, frames_dir, temp_dir)
        if not denoise_result['success']:
            return denoise_result
        
        processed_frames_dir = denoise_result['processed_frames_dir']
        
        # 步骤3: 将处理后的图片重新拼接为视频
        print("Step 3: Reconstructing video from processed frames...")
        reconstruct_result = reconstruct_video_from_frames(
            processed_frames_dir, 
            frames_info, 
            temp_dir
        )
        
        return reconstruct_result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def split_video_to_frames(video_path, temp_dir):
    """
    将视频分割为256x256的图片片段
    """
    try:
        # 创建帧存储目录
        frames_dir = os.path.join(temp_dir, 'original_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Cannot open video file'}
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")
        
        # 计算网格分割
        tile_size = 256
        cols = (original_width + tile_size - 1) // tile_size  # 向上取整
        rows = (original_height + tile_size - 1) // tile_size
        
        frames_info = {
            'fps': fps,
            'total_frames': total_frames,
            'original_size': (original_width, original_height),
            'tile_size': tile_size,
            'grid_size': (cols, rows),
            'frame_files': []
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 分割帧为256x256的tiles
            frame_tiles = []
            for row in range(rows):
                for col in range(cols):
                    # 计算tile位置（处理边界情况）
                    start_x = col * tile_size
                    start_y = row * tile_size
                    end_x = min(start_x + tile_size, original_width)
                    end_y = min(start_y + tile_size, original_height)
                    
                    # 提取tile
                    tile = frame_rgb[start_y:end_y, start_x:end_x]
                    
                    # 如果tile尺寸不足256x256，进行填充
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded_tile
                    
                    # 保存tile
                    tile_filename = f"frame_{frame_count:06d}_tile_{row}_{col}.png"
                    tile_path = os.path.join(frames_dir, tile_filename)
                    cv2.imwrite(tile_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                    
                    frame_tiles.append({
                        'filename': tile_filename,
                        'path': tile_path,
                        'position': (row, col),
                        'original_size': (end_y - start_y, end_x - start_x)  # 实际尺寸（可能小于256x256）
                    })
            
            frames_info['frame_files'].append(frame_tiles)
            frame_count += 1
            
            if frame_count % 30 == 0:  # 每30帧打印进度
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        print(f"✓ Video split into {frame_count} frames, {cols}x{rows} tiles per frame")
        
        return {
            'success': True,
            'frames_info': frames_info,
            'frames_dir': frames_dir,
            'processed_frames': frame_count
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Video splitting failed: {str(e)}'}

def denoise_frames(frames_info, frames_dir, temp_dir):
    """
    对每个图片片段进行降噪处理
    """
    try:
        processed_frames_dir = os.path.join(temp_dir, 'processed_frames')
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        total_tiles = sum(len(frame) for frame in frames_info['frame_files'])
        processed_tiles = 0
        
        # 处理每个tile
        for frame_index, frame_tiles in enumerate(frames_info['frame_files']):
            for tile_info in frame_tiles:
                original_path = tile_info['path']
                
                # 生成处理后的文件名
                processed_filename = f"processed_{tile_info['filename']}"
                processed_path = os.path.join(processed_frames_dir, processed_filename)
                
                # 使用现有的图片处理函数进行降噪
                process_success = process_image(original_path, processed_path)
                
                if process_success:
                    tile_info['processed_path'] = processed_path
                else:
                    # 如果处理失败，复制原图
                    shutil.copy2(original_path, processed_path)
                    tile_info['processed_path'] = processed_path
                    print(f"Warning: Processing failed for {tile_info['filename']}, using original")
                
                processed_tiles += 1
                if processed_tiles % 100 == 0:
                    print(f"Denoised {processed_tiles}/{total_tiles} tiles")
        
        print(f"✓ Denoising completed: {processed_tiles} tiles processed")
        
        return {
            'success': True,
            'processed_frames_dir': processed_frames_dir
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Frame denoising failed: {str(e)}'}

def reconstruct_video_from_frames(processed_frames_dir, frames_info, temp_dir):
    """
    将处理后的图片重新拼接为视频
    """
    try:
        output_video_path = os.path.join(temp_dir, 'processed_video.mp4')
        
        original_width, original_height = frames_info['original_size']
        fps = frames_info['fps']
        cols, rows = frames_info['grid_size']
        tile_size = frames_info['tile_size']
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))
        
        # 重建每一帧
        for frame_index, frame_tiles in enumerate(frames_info['frame_files']):
            # 创建空白画布
            reconstructed_frame = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # 将tiles拼接回原帧
            for tile_info in frame_tiles:
                row, col = tile_info['position']
                tile_height, tile_width = tile_info['original_size']
                
                # 读取处理后的tile
                processed_tile = cv2.imread(tile_info['processed_path'])
                processed_tile = cv2.cvtColor(processed_tile, cv2.COLOR_BGR2RGB)
                
                # 计算在重建帧中的位置
                start_x = col * tile_size
                start_y = row * tile_size
                end_x = start_x + tile_width
                end_y = start_y + tile_height
                
                # 将tile放回原位置（只取实际尺寸部分，去掉填充）
                reconstructed_frame[start_y:end_y, start_x:end_x] = processed_tile[:tile_height, :tile_width]
            
            # 将帧写入视频（转换回BGR）
            out.write(cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR))
            
            if (frame_index + 1) % 30 == 0:
                print(f"Reconstructed {frame_index + 1}/{len(frames_info['frame_files'])} frames")
        
        out.release()
        
        print(f"✓ Video reconstruction completed: {output_video_path}")
        
        return {
            'success': True,
            'processed_video_path': output_video_path,
            'original_size': f"{original_width}x{original_height}",
            'fps': fps,
            'total_frames': len(frames_info['frame_files'])
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Video reconstruction failed: {str(e)}'}
    
@api_bp.route('/fpga_single_inference', methods=['POST'])
def fpga_single_inference_demo():
    clean_results()
    try:
        if 'image' not in request.files:
            return jsonify(error="No image file provided"), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify(error="No selected file"), 400
        
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{timestamp}_{unique_id}{ext}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            processed_filename = f"processed_{name}_{timestamp}_{unique_id}.png"
            processed_filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename)
            
           #process_success = fpga_process_image(filepath, processed_filepath)
            process_success = fpga_process_image(
                image_path=filepath,
                output_path=processed_filepath,
                xmodel_path="models/Color/QAT_C_V1.xmodel"
            )
    
            result = {
                "status": "success",
                "message": "Image uploaded and processed successfully",
                "original_filename": original_filename,
                "saved_filename": filename,
                "filepath": filepath,
                "inference_result": {
                    "class": "processed_image",
                    "confidence": 0.95
                }
            }
            
            if process_success:
                result["processed_image_url"] = f"http://{request.host}/api/processed/{processed_filename}"
            
            return jsonify(result)
        else:
            return jsonify(error="File type not allowed"), 400
        
    except Exception as e:
        return jsonify(error=f"Failed to process image: {str(e)}"), 500
    


@api_bp.route('/fpga_inference_multiple', methods=['POST'])
def fpga_inference_multiple():
    try:
        if 'images' not in request.files:
            return jsonify(error="No image files provided"), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify(error="No selected files"), 400
        
        # get config
        upload_folder = current_app.config['UPLOAD_FOLDER']
        processed_folder = current_app.config['PROCESSED_FOLDER']
        host = request.host

        # initialize multiple FPGA DPU runners
        xmodel_path = "models/Color/QAT_C_V1.xmodel"
        g = xir.Graph.deserialize(xmodel_path)
        subgraphs = get_child_subgraph_dpu(g)

        # create multiple DPU runners for parallel processing
        num_runners = min(4, len(files))  # 最多4个runner，不超过文件数
        dpu_runners = [vart.Runner.create_runner(subgraphs[0], "run") for _ in range(num_runners)]

        # get model parameters (all runner parameters are the same)
        inputTensors = dpu_runners[0].get_input_tensors()
        outputTensors = dpu_runners[0].get_output_tensors()
        input_ndim = tuple(inputTensors[0].dims)
        output_ndim = tuple(outputTensors[0].dims)
        
        input_fixpos = inputTensors[0].get_attr("fix_point")
        input_scale = 2**input_fixpos
        output_fixpos = outputTensors[0].get_attr("fix_point")
        output_scale = 2**output_fixpos if output_fixpos is not None else 1.0
        
        results = []
        valid_files = 0
        
        # use ThreadPoolExecutor for parallel processing

        
        def process_single_file(file_info):
            file_idx, file = file_info
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                name, ext = os.path.splitext(original_filename)
                
                # save uploaded file
                upload_filename = f"{name}_{timestamp}_{unique_id}{ext}"
                upload_filepath = os.path.join(upload_folder, upload_filename)
                file.save(upload_filepath)
                
                # processed image path
                processed_filename = f"processed_{name}_{timestamp}_{unique_id}.png"
                processed_filepath = os.path.join(processed_folder, processed_filename)
                
                # assign DPU runners
                runner_idx = file_idx % num_runners
                success = fpga_single_inference(
                    dpu_runners[runner_idx], 
                    upload_filepath, 
                    processed_filepath,
                    input_ndim,
                    output_ndim,
                    input_scale,
                    output_scale
                )
                
                file_result = {
                    "original_filename": original_filename,
                    "saved_filename": upload_filename,
                    "filepath": upload_filepath,
                    "inference_result": {
                        "class": "processed_image",
                        "confidence": 0.95,
                        "status": "success" if success else "failed"
                    }
                }
                
                if success:
                    file_result["processed_image_url"] = f"http://{host}/api/processed/{processed_filename}"
                
                return file_result
            return None
        
        # parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_runners) as executor:
            file_infos = [(i, file) for i, file in enumerate(files)]
            processed_results = list(executor.map(process_single_file, file_infos))
        
        # filter out None results
        results = [result for result in processed_results if result is not None]
        valid_files = len(results)
        
        # clean up runners
        for runner in dpu_runners:
            del runner
        
        if valid_files == 0:
            return jsonify(error="No valid image files found"), 400
        
        response_data = {
            "status": "success",
            "message": f"Processed {valid_files} images successfully using {num_runners} FPGA cores",
            "total_images": valid_files,
            "fpga_cores_used": num_runners,
            "results": results
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify(error=f"Failed to process images with FPGA: {str(e)}"), 500

def fpga_single_inference(dpu_runner, input_path, output_path, input_ndim, output_ndim, input_scale, output_scale):
    """get dpu subgraph from xir graph"""
    try:
        # read and preprocess image
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            return False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_height, input_width = input_ndim[1:3]
        
        if image_rgb.shape[:2] != (input_height, input_width):
            image_resized = cv2.resize(image_rgb, (input_width, input_height))
        else:
            image_resized = image_rgb

        # preprocess
        processed_image = image_resized.astype(np.float32) * (1.0 / 255.0) * input_scale
        processed_image = processed_image.astype(np.int8)
        
        # pre-process
        input_data = [np.empty(input_ndim, dtype=np.int8, order="C")]
        output_data = [np.empty(output_ndim, dtype=np.int8, order="C")]
        
        input_data[0][0, ...] = processed_image.reshape(input_ndim[1:])
        
        # inference
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        
        # post-process
        output_img = output_data[0][0]
        denoised_float = (output_img.astype(np.float32) / output_scale) * 255.0
        denoised_float = np.clip(denoised_float, 0, 255).astype(np.uint8)
        denoised_bgr = cv2.cvtColor(denoised_float, cv2.COLOR_RGB2BGR)
        
        # save result
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, denoised_bgr)
        return True
        
    except Exception as e:
        print(f"FPGA inference failed for {input_path}: {e}")
        return False

def get_child_subgraph_dpu(graph):
    """get dpu subgraph from xir graph"""
    root_subgraph = graph.get_root_subgraph()
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]