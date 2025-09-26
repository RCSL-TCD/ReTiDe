from PIL import Image
import sys
import os

def check_image_mode(image_path):
    # 检查文件是否存在
    if not os.path.isfile(image_path):
        print(f"文件不存在: {image_path}")
        return

    try:
        with Image.open(image_path) as img:
            mode = img.mode
            if mode == 'L':
                print(f"{image_path} 是灰度图（Grayscale）")
            elif mode == 'RGB':
                print(f"{image_path} 是RGB图")
            elif mode == 'RGBA':
                print(f"{image_path} 是RGBA图（含透明通道）")
            else:
                print(f"{image_path} 的模式为 {mode}，不是标准灰度图或RGB图")
    except Exception as e:
        print(f"无法打开图片: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_image_mode.py <图片路径>")
    else:
        image_path = sys.argv[1]
        check_image_mode(image_path)
