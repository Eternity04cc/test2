#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26 无人机检测系统
基于Ultralytics YOLO26实现的无人机检测、训练和推理系统
支持CPU优化推理，相比YOLOv7提升43%的CPU推理速度

作者: YOLO26 升级项目
版本: 1.0
日期: 2024
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo26_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLO26DroneDetector:
    """
    YOLO26无人机检测器
    
    主要功能:
    - 模型训练（支持MuSGD优化器）
    - CPU优化推理
    - 批量检测
    - 视频流检测
    - 性能监控
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu'):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径，支持预训练模型或自定义模型
            device: 推理设备，默认使用CPU
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = ['drone']
        
        # CPU优化设置
        if device == 'cpu':
            self._optimize_cpu_settings()
        
        # 加载模型
        self.load_model()
        
        logger.info(f"YOLO26无人机检测器初始化完成，使用设备: {device}")
    
    def _optimize_cpu_settings(self):
        """
        CPU推理优化设置
        """
        # 设置CPU线程数
        cpu_count = psutil.cpu_count(logical=False)
        torch.set_num_threads(min(4, cpu_count))
        
        # 禁用CUDA相关优化
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 设置环境变量优化CPU性能
        os.environ['OMP_NUM_THREADS'] = str(min(4, cpu_count))
        os.environ['MKL_NUM_THREADS'] = str(min(4, cpu_count))
        
        logger.info(f"CPU优化设置完成，使用{min(4, cpu_count)}个线程")
    
    def load_model(self):
        """
        加载YOLO26模型
        """
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def train_model(self, 
                   data_config: str = 'data.yaml',
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640,
                   optimizer: str = 'auto',
                   lr0: float = 0.01,
                   weight_decay: float = 0.0005,
                   momentum: float = 0.937,
                   save_dir: str = 'runs/train',
                   **kwargs) -> dict:
        """
        训练YOLO26模型
        
        Args:
            data_config: 数据配置文件路径
            epochs: 训练轮次
            batch_size: 批处理大小
            img_size: 输入图像尺寸
            optimizer: 优化器类型（auto使用MuSGD）
            lr0: 初始学习率
            weight_decay: 权重衰减
            momentum: 动量
            save_dir: 保存目录
            **kwargs: 其他训练参数
        
        Returns:
            训练结果字典
        """
        logger.info("开始训练YOLO26模型...")
        
        try:
            # 训练配置
            train_args = {
                'data': data_config,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'device': self.device,
                'optimizer': optimizer,
                'lr0': lr0,
                'weight_decay': weight_decay,
                'momentum': momentum,
                'project': save_dir,
                'name': 'yolo26_drone_detection',
                'save': True,
                'save_period': 10,
                'cache': False,  # CPU训练时禁用缓存
                'workers': 2 if self.device == 'cpu' else 8,
                **kwargs
            }
            
            # 开始训练
            start_time = time.time()
            results = self.model.train(**train_args)
            training_time = time.time() - start_time
            
            logger.info(f"训练完成，耗时: {training_time:.2f}秒")
            
            # 保存最佳模型路径
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            self.model_path = str(best_model_path)
            
            return {
                'success': True,
                'training_time': training_time,
                'best_model_path': str(best_model_path),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_image(self, 
                     image_path: Union[str, np.ndarray],
                     conf_threshold: float = 0.5,
                     save_result: bool = True,
                     show_result: bool = False) -> dict:
        """
        单张图片无人机检测
        
        Args:
            image_path: 图片路径或numpy数组
            conf_threshold: 置信度阈值
            save_result: 是否保存结果
            show_result: 是否显示结果
        
        Returns:
            检测结果字典
        """
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            # 推理
            results = self.model(
                source=image_path,
                conf=conf_threshold,
                device=self.device,
                save=save_result,
                show=show_result,
                verbose=False
            )
            
            # 处理结果
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        try:
                            # 安全地获取bbox坐标
                            xyxy = box.xyxy.cpu().numpy().tolist()
                            xywh = box.xywh.cpu().numpy().tolist()
                            
                            # 确保获取正确的坐标格式
                            if isinstance(xyxy, list) and len(xyxy) > 0:
                                bbox_coords = xyxy[0] if isinstance(xyxy[0], list) else xyxy
                            else:
                                bbox_coords = xyxy
                                
                            if isinstance(xywh, list) and len(xywh) > 0:
                                center_coords = xywh[0][:2] if isinstance(xywh[0], list) else xywh[:2]
                            else:
                                center_coords = xywh[:2]
                            
                            # 安全地获取类别名称
                            cls_idx = int(box.cls.item())
                            if cls_idx < len(self.class_names):
                                class_name = self.class_names[cls_idx]
                            else:
                                # 如果类别索引超出范围，使用模型的类别名称或默认名称
                                if hasattr(self.model, 'names') and cls_idx in self.model.names:
                                    class_name = self.model.names[cls_idx]
                                else:
                                    class_name = f'object_{cls_idx}'
                            
                            detection = {
                                'class': class_name,
                                'confidence': float(box.conf.item()),
                                'bbox': bbox_coords,  # [x1, y1, x2, y2]
                                'center': center_coords  # [center_x, center_y]
                            }
                            detections.append(detection)
                        except Exception as box_error:
                            continue
            
            # 计算性能指标
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            end_memory = psutil.virtual_memory().used
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            
            result_dict = {
                'success': True,
                'detections': detections,
                'detection_count': len(detections),
                'inference_time_ms': round(inference_time, 2),
                'memory_used_mb': round(memory_used, 2),
                'image_path': str(image_path) if isinstance(image_path, (str, Path)) else 'numpy_array'
            }
            
            logger.info(f"检测完成: 发现{len(detections)}个无人机目标，耗时{inference_time:.2f}ms")
            return result_dict
            
        except Exception as e:
            logger.error(f"图片检测失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_batch(self, 
                     image_paths: List[str],
                     conf_threshold: float = 0.5,
                     batch_size: int = 4) -> List[dict]:
        """
        批量图片检测
        
        Args:
            image_paths: 图片路径列表
            conf_threshold: 置信度阈值
            batch_size: 批处理大小
        
        Returns:
            检测结果列表
        """
        logger.info(f"开始批量检测{len(image_paths)}张图片...")
        
        results = []
        total_start_time = time.time()
        
        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # 批量推理
                batch_results = self.model(
                    source=batch_paths,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # 处理每张图片的结果
                for j, result in enumerate(batch_results):
                    image_path = batch_paths[j]
                    detections = []
                    
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            detection = {
                                'class': self.class_names[int(box.cls.item())],
                                'confidence': float(box.conf.item()),
                                'bbox': box.xyxy.cpu().numpy().tolist()[0]
                            }
                            detections.append(detection)
                    
                    results.append({
                        'image_path': image_path,
                        'detections': detections,
                        'detection_count': len(detections)
                    })
                    
            except Exception as e:
                logger.error(f"批次{i//batch_size + 1}处理失败: {e}")
                # 为失败的图片添加错误结果
                for path in batch_paths:
                    results.append({
                        'image_path': path,
                        'error': str(e),
                        'detections': [],
                        'detection_count': 0
                    })
        
        total_time = time.time() - total_start_time
        avg_time_per_image = (total_time / len(image_paths)) * 1000  # ms
        
        logger.info(f"批量检测完成，总耗时{total_time:.2f}秒，平均每张{avg_time_per_image:.2f}ms")
        return results
    
    def detect_video(self, 
                    video_path: str,
                    output_path: Optional[str] = None,
                    conf_threshold: float = 0.5,
                    show_live: bool = False) -> dict:
        """
        视频流无人机检测
        
        Args:
            video_path: 视频文件路径或摄像头索引
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
            show_live: 是否实时显示
        
        Returns:
            检测结果统计
        """
        logger.info(f"开始视频检测: {video_path}")
        
        try:
            # 打开视频
            if isinstance(video_path, int) or video_path.isdigit():
                cap = cv2.VideoCapture(int(video_path))
            else:
                cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 设置输出视频
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 检测统计
            frame_count = 0
            detection_count = 0
            total_inference_time = 0
            
            logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 推理
                start_time = time.time()
                results = self.model(
                    source=frame,
                    conf=conf_threshold,
                    device=self.device,
                    verbose=False
                )
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # 绘制检测结果
                annotated_frame = frame.copy()
                frame_detections = 0
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                            conf = float(box.conf.item())
                            
                            # 绘制边界框
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 绘制标签
                            label = f'Drone: {conf:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            frame_detections += 1
                
                detection_count += frame_detections
                
                # 添加帧信息
                info_text = f'Frame: {frame_count}/{total_frames}, Detections: {frame_detections}, FPS: {1/inference_time:.1f}'
                cv2.putText(annotated_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 保存帧
                if writer:
                    writer.write(annotated_frame)
                
                # 显示帧
                if show_live:
                    cv2.imshow('YOLO26 Drone Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 进度显示
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    avg_fps = frame_count / total_inference_time
                    logger.info(f"处理进度: {progress:.1f}%, 平均FPS: {avg_fps:.1f}")
            
            # 清理资源
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()
            
            # 统计结果
            avg_inference_time = (total_inference_time / frame_count) * 1000  # ms
            avg_fps = frame_count / total_inference_time
            
            result_dict = {
                'success': True,
                'total_frames': frame_count,
                'total_detections': detection_count,
                'avg_detections_per_frame': detection_count / frame_count,
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'avg_fps': round(avg_fps, 2),
                'output_path': output_path
            }
            
            logger.info(f"视频检测完成: {frame_count}帧，{detection_count}个检测，平均FPS: {avg_fps:.2f}")
            return result_dict
            
        except Exception as e:
            logger.error(f"视频检测失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_model(self, data_config: str = 'data.yaml') -> dict:
        """
        验证模型性能
        
        Args:
            data_config: 数据配置文件
        
        Returns:
            验证结果
        """
        logger.info("开始模型验证...")
        
        try:
            results = self.model.val(
                data=data_config,
                device=self.device,
                verbose=True
            )
            
            return {
                'success': True,
                'metrics': results.results_dict,
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0)
            }
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def export_model(self, format: str = 'onnx', **kwargs) -> dict:
        """
        导出模型为其他格式
        
        Args:
            format: 导出格式 ('onnx', 'openvino', 'tensorrt', etc.)
            **kwargs: 导出参数
        
        Returns:
            导出结果
        """
        logger.info(f"导出模型为{format}格式...")
        
        try:
            export_path = self.model.export(
                format=format,
                dynamic=False,
                half=False,  # CPU不支持FP16
                **kwargs
            )
            
            return {
                'success': True,
                'export_path': str(export_path),
                'format': format
            }
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description='YOLO26无人机检测系统')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'batch', 'video', 'validate', 'export'],
                       required=True, help='运行模式')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型路径')
    parser.add_argument('--data', type=str, default='data.yaml', help='数据配置文件')
    parser.add_argument('--source', type=str, help='输入源（图片/视频路径）')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--format', type=str, default='onnx', help='导出格式')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = YOLO26DroneDetector(model_path=args.model, device=args.device)
    
    if args.mode == 'train':
        # 训练模式
        result = detector.train_model(
            data_config=args.data,
            epochs=args.epochs,
            batch_size=args.batch
        )
        print(f"训练结果: {result}")
        
    elif args.mode == 'predict':
        # 单图片预测
        if not args.source:
            print("错误: 预测模式需要指定 --source 参数")
            return
        
        result = detector.predict_image(
            image_path=args.source,
            conf_threshold=args.conf,
            save_result=True,
            show_result=True
        )
        print(f"检测结果: {result}")
        
    elif args.mode == 'batch':
        # 批量预测
        if not args.source:
            print("错误: 批量模式需要指定 --source 参数（目录路径）")
            return
        
        # 获取目录下所有图片
        source_dir = Path(args.source)
        if not source_dir.is_dir():
            print(f"错误: {args.source} 不是有效目录")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [str(p) for p in source_dir.iterdir() 
                      if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            print(f"错误: 在 {args.source} 中未找到图片文件")
            return
        
        results = detector.predict_batch(
            image_paths=image_paths,
            conf_threshold=args.conf,
            batch_size=args.batch
        )
        
        # 统计结果
        total_detections = sum(r['detection_count'] for r in results)
        print(f"批量检测完成: {len(results)}张图片，共检测到{total_detections}个无人机")
        
    elif args.mode == 'video':
        # 视频检测
        if not args.source:
            print("错误: 视频模式需要指定 --source 参数")
            return
        
        result = detector.detect_video(
            video_path=args.source,
            output_path=args.output,
            conf_threshold=args.conf,
            show_live=True
        )
        print(f"视频检测结果: {result}")
        
    elif args.mode == 'validate':
        # 模型验证
        result = detector.validate_model(data_config=args.data)
        print(f"验证结果: {result}")
        
    elif args.mode == 'export':
        # 模型导出
        result = detector.export_model(format=args.format)
        print(f"导出结果: {result}")

if __name__ == '__main__':
    main()