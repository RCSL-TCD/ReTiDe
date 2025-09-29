import os
import numpy as np
from moviepy.editor import VideoFileClip

def add_white_noise_to_frame(frame, noise_level=35):
    """
    为视频帧添加白噪声
    noise_level: 噪声等级，控制噪声的强度 (0-100)
    """
    # 将帧转换为浮点数格式以便处理
    frame_float = frame.astype(np.float32)
    
    # 计算噪声强度 (基于像素值范围0-255和噪声等级)
    max_noise = 255 * (noise_level / 100.0)
    
    # 生成与帧相同形状的白噪声
    white_noise = np.random.normal(0, max_noise, frame.shape)
    
    # 将噪声添加到帧
    noisy_frame = frame_float + white_noise
    
    # 确保像素值在0-255范围内
    noisy_frame = np.clip(noisy_frame, 0, 255)
    
    # 转换回uint8格式
    noisy_frame = noisy_frame.astype(np.uint8)
    
    return noisy_frame

def process_video_frames(get_frame, t, noise_level=35):
    """
    处理视频帧的函数，用于moviepy的fl_image
    """
    frame = get_frame(t)
    return add_white_noise_to_frame(frame, noise_level)

def process_video(input_file, output_dir="output"):
    """
    处理视频：裁剪10s-20s片段，并生成带图像噪声的版本
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return
    
    try:
        # 加载视频文件
        print("正在加载视频文件...")
        video = VideoFileClip(input_file)
        
        # 获取视频总时长
        total_duration = video.duration
        print(f"视频总时长: {total_duration:.2f}秒")
        
        # 检查裁剪时间段是否有效
        if total_duration < 20:
            print(f"错误：视频时长不足20秒，无法裁剪10-20秒片段")
            return
        
        # 裁剪10-20秒片段
        print("正在裁剪10-20秒片段...")
        cropped_video = video.subclip(10, 20)
        
        # 保存原始裁剪片段
        original_output = os.path.join(output_dir, "cropped_original.mp4")
        print("正在保存原始裁剪片段...")
        cropped_video.write_videofile(
            original_output,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # 为视频帧添加白噪声
        print("正在处理视频帧：添加白噪声...")
        
        # 使用fl_image对每一帧应用噪声处理
        noisy_video = cropped_video.fl_image(
            lambda frame: add_white_noise_to_frame(frame, noise_level=35)
        )
        
        # 保存带图像噪声的视频片段
        noisy_output = os.path.join(output_dir, "cropped_noisy.mp4")
        print("正在保存带图像噪声的视频片段...")
        noisy_video.write_videofile(
            noisy_output,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # 关闭所有剪辑以释放内存
        video.close()
        cropped_video.close()
        noisy_video.close()
        
        print(f"\n处理完成！")
        print(f"原始裁剪片段: {original_output}")
        print(f"带图像噪声片段: {noisy_output}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    # 设置输入文件路径
    input_video = "cai.mp4"
    
    # 处理视频
    process_video(input_video)
    
    print("所有任务完成！")