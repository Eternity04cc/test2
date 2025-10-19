#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机检测YOLO26模型调用脚本
用于Node.js后端通过子进程调用
"""

import sys
import json
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import logging
    import warnings
    # 禁用所有日志输出到stdout，避免干扰JSON输出
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger('yolo26_drone_detection').setLevel(logging.CRITICAL)
    logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    
    # 重定向stdout到stderr，避免YOLO输出干扰JSON
    import contextlib
    
    from yolo26_drone_detection import YOLO26DroneDetector
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"无法导入YOLO26模块: {str(e)}",
        "detections": [],
        "droneCount": 0,
        "timestamp": time.time() * 1000
    }))
    sys.exit(1)

def process_image(image_data, confidence_threshold=0.5):
    """
    处理图片并进行无人机检测
    
    Args:
        image_data: base64编码的图片数据或图片文件路径
        confidence_threshold: 置信度阈值
    
    Returns:
        dict: 检测结果
    """
    try:
        # 初始化检测器，使用项目中的yolov8n.pt模型
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'yolov8n.pt')
        detector = YOLO26DroneDetector(model_path=model_path)
        
        # 处理输入数据
        if image_data.startswith('data:image'):
            # 处理base64图片数据
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes))
        elif os.path.isfile(image_data):
            # 处理文件路径
            image = Image.open(image_data)
        else:
            raise ValueError("无效的图片数据格式")
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 进行检测，重定向stdout避免干扰JSON输出
        with contextlib.redirect_stdout(sys.stderr):
            result = detector.predict_image(image_array, confidence_threshold)
        
        # 检查检测结果
        if not result.get('success', False):
            return result
        
        # 转换检测结果格式
        detections = []
        raw_detections = result.get('detections', [])
        
        for detection in raw_detections:
            if 'bbox' in detection and detection['bbox']:
                bbox = detection['bbox']
                
                # 确保bbox是列表且有4个元素
                if isinstance(bbox, list) and len(bbox) >= 4:
                    try:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        formatted_detection = {
                            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, width, height]
                            "x": int((x1 + x2) / 2),  # 中心点x坐标
                            "y": int((y1 + y2) / 2),  # 中心点y坐标
                            "confidence": detection.get('confidence', 0.0),
                            "class": detection.get('class', 'drone')
                        }
                        detections.append(formatted_detection)
                    except Exception as e:
                        continue
        
        # 返回结果
        result = {
            "success": True,
            "detections": detections,
            "droneCount": len(detections),
            "inferenceTime": result.get('inference_time_ms', 0),
            "timestamp": time.time() * 1000,
            "imageSize": {
                "width": image.width,
                "height": image.height
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "detections": [],
            "droneCount": 0,
            "timestamp": time.time() * 1000
        }

def main():
    """
    主函数，处理命令行参数
    """
    try:
        if len(sys.argv) < 2:
            raise ValueError("缺少必要参数")
        
        # 获取命令行参数
        command = sys.argv[1]
        
        if command == "detect":
            if len(sys.argv) < 3:
                raise ValueError("缺少图片数据参数")
            
            image_data = sys.argv[2]
            confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            
            # 进行检测
            result = process_image(image_data, confidence_threshold)
            
        elif command == "test":
            # 测试模式
            result = {
                "success": True,
                "message": "YOLO26检测器测试成功",
                "timestamp": time.time() * 1000
            }
            
        else:
            raise ValueError(f"未知命令: {command}")
        
        # 输出JSON结果
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": time.time() * 1000
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()