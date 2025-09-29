import requests
import json
import os
import sys
from datetime import datetime
import shutil

from configs.server_config import server_ip, port

def test_api():
    url = f"http://{server_ip}:{port}/api/hello"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("ok:", response.json())
        else:
            print("ng:", response.status_code, response.text)
    except Exception as e:
        print("ng:", e)

def single_f32_inference(image_path):
    url = f"http://{server_ip}:{port}/api/f32_inference"
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/images', exist_ok=True)
    
    try:
        # stage1: upload image
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
        # stage2: handle response
        if response.status_code == 200:
            try:
                result = response.json()
            except json.JSONDecodeError:
                print(f"✗ Returned json is not valid: {response.text}")
                return False
            
            if not isinstance(result, dict):
                print(f"✗ Returned type is not dict: {type(result)}")
                print(f"return result: {result}")
                return False
            
            print(f"✓ upload ok")
            
            saved_filename = result.get('saved_filename', 'unkown filename')
            print(f"save on server: {saved_filename}")
            
            inference = result.get('inference_result', {})
            if isinstance(inference, dict):
                class_name = inference.get('class', 'N/A')
                confidence = inference.get('confidence', 0)
                print(f"  prediction result: {class_name} (confidence: {confidence:.2f})")
            else:
                print(f"  prediction result: {inference}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            processed_image_saved = False
            processed_image_path = None
            
            if 'processed_image_url' in result and result['processed_image_url']:
                processed_image_url = result['processed_image_url']
                processed_image_path = f"results/images/processed_{image_name}_{timestamp}.png"
                
                img_response = requests.get(processed_image_url)
                if img_response.status_code == 200:
                    with open(processed_image_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"✓ processed image saved: {processed_image_path}")
                    processed_image_saved = True
                else:
                    print(f"✗ failed to download processed image: {img_response.status_code}")

            original_copy_path = f"results/images/original_{image_name}_{timestamp}.png"
            try:
                shutil.copy2(image_path, original_copy_path)
                print(f"✓ original copy saved: {original_copy_path}")
            except Exception as e:
                print(f"✗ failed to save original copy: {e}")

            result_file = f"results/result_{image_name}_{timestamp}.json"
            save_data = {
                "original_image": image_path,
                "original_copy": original_copy_path,
                "upload_time": datetime.now().isoformat(),
                "server_response": result,
                "local_saved_files": {
                    "original_copy": original_copy_path,
                    "processed_image": processed_image_path if processed_image_saved else None,
                    "result_json": result_file
                }
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"✓ complete result saved: {result_file}")
            return True
            
        else:
            print(f"✗ upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ error: {e}")
        import traceback
        traceback.print_exc() 
        return False
def multiple_f32_inference(folder_path):
    url = f"http://{server_ip}:{port}/api/f32_inference_multiple"
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/images', exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                image_files.append(file_path)
    
    if not image_files:
        print(f"✗ cant find images in folder: {folder_path} ")
        return False
    
    print(f"✓ found {len(image_files)} images")
    
    try:
        files = []
        for image_path in image_files:
            files.append(('images', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))
        
        response = requests.post(url, files=files)
        
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            try:
                result = response.json()
            except json.JSONDecodeError:
                print(f"✗ return json invalid: {response.text}")
                return False
            
            if not isinstance(result, dict):
                print(f"✗ return not dict: {type(result)}")
                print(f"result: {result}")
                return False
            
            print(f"✓ batch upload ok")
            print(f"found {result.get('total_images', 0)} images")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = os.path.basename(os.path.normpath(folder_path))
            batch_results = []
            
            for i, file_result in enumerate(result.get('results', [])):
                print(f"\n--- image {i+1} ---")
                
                saved_filename = file_result.get('saved_filename', 'unkown filename')
                print(f"server save path: {saved_filename}")

                inference = file_result.get('inference_result', {})
                if isinstance(inference, dict):
                    class_name = inference.get('class', 'N/A')
                    confidence = inference.get('confidence', 0)
                    print(f"denoise result: {class_name}")
                else:
                    print(f"denoise result: {inference}")
                
                original_filename = file_result.get('original_filename', '')
                image_name = os.path.splitext(original_filename)[0]
                
                processed_image_saved = False
                processed_image_path = None
                
                if 'processed_image_url' in file_result and file_result['processed_image_url']:
                    processed_image_url = file_result['processed_image_url']
                    processed_image_path = f"results/images/processed_{image_name}_{timestamp}.png"
                    
                    img_response = requests.get(processed_image_url)
                    if img_response.status_code == 200:
                        with open(processed_image_path, 'wb') as f:
                            f.write(img_response.content)
                        print(f"✓ processed image saved: {processed_image_path}")
                        processed_image_saved = True
                    else:
                        print(f"✗ download processed image failed: {img_response.status_code}")

                original_file_path = None
                for img_file in image_files:
                    if os.path.basename(img_file) == original_filename:
                        original_file_path = img_file
                        break
                
                original_copy_path = None
                if original_file_path:
                    original_copy_path = f"results/images/original_{image_name}_{timestamp}.png"
                    try:
                        shutil.copy2(original_file_path, original_copy_path)
                        print(f"✓ original image copy saved: {original_copy_path}")
                    except Exception as e:
                        print(f"✗ save original image copy failed: {e}")


                single_result = {
                    "original_image": original_file_path,
                    "original_copy": original_copy_path,
                    "upload_time": datetime.now().isoformat(),
                    "server_response": file_result,
                    "local_saved_files": {
                        "original_copy": original_copy_path,
                        "processed_image": processed_image_path if processed_image_saved else None
                    }
                }
                batch_results.append(single_result)
            
            result_file = f"results/batch_result_{folder_name}_{timestamp}.json"
            save_data = {
                "batch_info": {
                    "folder_path": folder_path,
                    "total_images": len(batch_results),
                    "processed_time": datetime.now().isoformat(),
                    "timestamp": timestamp
                },
                "individual_results": batch_results
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"\n✓ batch process complete, results saved: {result_file}")
            return True
            
        else:
            print(f"✗ batch upload failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ error: {e}")
        import traceback
        traceback.print_exc()
        return False

def video_denoise_inference(video_path):
    """
    客户端调用视频降噪接口
    """
    url = f"http://{server_ip}:{port}/api/video_denoise"
    
    os.makedirs('results/videos', exist_ok=True)
    
    if not os.path.isfile(video_path) or not video_path.lower().endswith('.mp4'):
        print(f"✗ Invalid MP4 video file: {video_path}")
        return False
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
            
            print(f"✓ Uploading video: {os.path.basename(video_path)}")
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            # 保存处理后的视频
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"results/videos/denoised_{video_name}_{timestamp}.mp4"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Video processing completed")
            print(f"✓ Processed video saved: {output_path}")
            
            # 保存处理记录
            result_file = f"results/video_result_{video_name}_{timestamp}.json"
            save_data = {
                "original_video": video_path,
                "processed_video": output_path,
                "processed_time": datetime.now().isoformat(),
                "timestamp": timestamp
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results saved: {result_file}")
            return True
            
        else:
            try:
                error_info = response.json()
                print(f"✗ Video processing failed: {error_info.get('error', 'Unknown error')}")
            except:
                print(f"✗ Video processing failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False