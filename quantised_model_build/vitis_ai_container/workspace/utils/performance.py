import time
import torch
import numpy as np
from thop import profile  # 用于计算FLOPs/GOPs
import pynvml  # 用于GPU功耗测量（仅NVIDIA GPU）
import psutil  # 用于CPU功耗估算
from models.my_models_0708 import UnetGenerator_hardware_pixelshuffle as Unet_P
from models.my_models_0708 import UnetGenerator_hardware as Unet

unet_f32_weight_path = 'models/unet_f32.pt'
unet_p_f32_weight_path = 'models/unet_p_f32.pt'

def measure_model_performance(model, input_shape=(1, 3, 256, 256), device='cuda', num_runs=100, warmup=10):
    """
    测量模型在指定设备上的性能指标
    
    参数:
    model: PyTorch模型
    input_shape: 输入张量形状 (batch, channels, height, width)
    device: 'cuda' 或 'cpu'
    num_runs: 测量运行次数
    warmup: 预热运行次数
    
    返回:
    metrics: 包含性能指标的字典
    """
    # 设置设备
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    inputs = torch.randn(input_shape).to(device)
    
    # 计算FLOPs和参数数量
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    gops = flops / 1e9  # 转换为GOPs (Giga Operations)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(inputs)
    
    # 测量延迟和吞吐量
    latencies = []
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_run = time.time()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保CUDA操作完成
                
            end_run = time.time()
            latencies.append(end_run - start_run)
    
    total_time = time.time() - start_time
    
    # 计算指标
    latency_avg = np.mean(latencies) * 1000  # 转换为毫秒
    latency_min = np.min(latencies) * 1000
    latency_max = np.max(latencies) * 1000
    throughput = num_runs / total_time  # 样本/秒
    
    # 计算GOP/s
    gops_per_sec = gops / (np.mean(latencies))
    
    # 功耗测量
    power_avg = 0
    if device.type == 'cuda':
        power_avg = measure_gpu_power(model, inputs, num_runs)
    else:
        power_avg = measure_cpu_power(model, inputs, num_runs)
    
    # 返回结果
    return {
        "device": device.type.upper(),
        "gops": gops,
        "gops_per_sec": gops_per_sec,
        "latency_avg_ms": latency_avg,
        "latency_min_ms": latency_min,
        "latency_max_ms": latency_max,
        "throughput_samples_per_sec": throughput,
        "power_avg_w": power_avg,
        "flops": flops,
        "params": params,
        "energy_per_inference_j": (power_avg * latency_avg / 1000) if power_avg > 0 else 0
    }

def measure_gpu_power(model, inputs, num_runs=50):
    """测量GPU平均功耗（仅支持NVIDIA GPU）"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        power_readings = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # 开始测量前读取一次
                pynvml.nvmlDeviceGetUtilizationRates(handle)
                power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                
                # 运行推理
                _ = model(inputs)
                torch.cuda.synchronize()
                
                # 结束后再读取一次
                power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_readings.append((power_start + power_end) / 2)
        
        pynvml.nvmlShutdown()
        return np.mean(power_readings)
    
    except Exception as e:
        print(f"GPU power measurement failed: {str(e)}")
        return 0

def measure_cpu_power(model, inputs, num_runs=50):
    """估算CPU平均功耗（基于系统级测量）"""
    try:
        # 获取初始功耗状态（如果可用）
        try:
            initial_power = psutil.sensors_battery().power_plugged
        except:
            initial_power = None
        
        power_readings = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # 运行推理
                start_time = time.time()
                _ = model(inputs)
                end_time = time.time()
                
                # 估算功耗（基于CPU利用率）
                cpu_percent = psutil.cpu_percent(interval=end_time - start_time)
                # 简单估算：假设最大TDP为65W，利用率线性相关
                estimated_power = 65 * (cpu_percent / 100) * 0.8  # 0.8是效率因子
                power_readings.append(estimated_power)
        
        return np.mean(power_readings)
    
    except Exception as e:
        print(f"CPU power measurement failed: {str(e)}")
        return 0

def print_performance_results(results):
    """格式化打印性能结果"""
    print("\n" + "="*70)
    print(f"Model Performance Report - {results['device']}")
    print("="*70)
    print(f"FLOPs: {results['flops'] / 1e9:.2f} GOPs")
    print(f"Parameters: {results['params'] / 1e6:.2f} Million")
    print(f"Avg Latency: {results['latency_avg_ms']:.2f} ms")
    print(f"Min Latency: {results['latency_min_ms']:.2f} ms")
    print(f"Max Latency: {results['latency_max_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"GOP/s: {results['gops_per_sec']:.2f}")
    print(f"Avg Power: {results['power_avg_w']:.2f} W")
    print(f"Energy per Inference: {results['energy_per_inference_j'] * 1000:.2f} mJ")
    print("="*70)

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = Unet_P(input_nc=3, output_nc=3, num_downs=8)
    model.load_state_dict(torch.load(unet_p_f32_weight_path))
    
    # 测量CPU性能
    print("\nMeasuring CPU Performance...")
    cpu_results = measure_model_performance(model, device='cpu')
    print_performance_results(cpu_results)
    
    # 测量GPU性能（如果可用）
    if torch.cuda.is_available():
        print("\nMeasuring GPU Performance...")
        gpu_results = measure_model_performance(model, device='cuda')
        print_performance_results(gpu_results)
    else:
        print("\nCUDA not available. Skipping GPU measurements.")
    
    # 比较结果
    if torch.cuda.is_available():
        print("\nPerformance Comparison:")
        print(f"Speedup (Latency): {cpu_results['latency_avg_ms'] / gpu_results['latency_avg_ms']:.2f}x")
        print(f"Throughput Ratio: {gpu_results['throughput_samples_per_sec'] / cpu_results['throughput_samples_per_sec']:.2f}x")
        print(f"GOP/s Ratio: {gpu_results['gops_per_sec'] / cpu_results['gops_per_sec']:.2f}x")
        print(f"Energy Efficiency Ratio: {cpu_results['energy_per_inference_j'] / gpu_results['energy_per_inference_j']:.2f}x")