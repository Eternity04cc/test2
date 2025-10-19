#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26 Drone Detection CPU优化推理脚本
支持单图片、批量检测和性能监控
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors
import psutil
import threading

class CPUOptimizedInference:
    """
    CPU优化的YOLO26推理器
    支持单图片、批量检测和性能监控
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化推理器
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = ['drone']  # 无人机类别
        
        # 性能监控
        self.inference_times = []
        self.cpu_usage = []
        self.memory_usage = []
        
        # CPU优化设置
        self._setup_cpu_optimization()
        
        # 加载模型
        self._load_model()
        
    def _setup_cpu_optimization(self):
        """
        设置CPU优化参数
        """
        # 设置CPU线程数
        num_threads = min(4, os.cpu_count())
        torch.set_num_threads(num_threads)
        
        # 启用CPU优化
        torch.backends.cudnn.enabled = False
        torch.backends.mkldnn.enabled = True
        
        # 设置环境变量
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        
        LOGGER.info(f"CPU优化设置完成 - 线程数: {num_threads}")
        
    def _load_model(self):
        """
        加载YOLO26模型
        """
        try:
            LOGGER.info(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 强制使用CPU
            self.model.to('cpu')
            
            # 预热模型
            self._warmup_model()
            
            LOGGER.info("模型加载完成")
            
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
            
    def _warmup_model(self, warmup_runs: int = 3):
        """
        预热模型以获得稳定的推理性能
        
        Args:
            warmup_runs: 预热运行次数
        """
        LOGGER.info("模型预热中...")
        
        # 创建虚拟输入
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for i in range(warmup_runs):
            try:
                _ = self.model.predict(
                    dummy_input,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device='cpu',
                    verbose=False
                )
            except Exception as e:
                LOGGER.warning(f"预热运行 {i+1} 失败: {e}")
                
        LOGGER.info("模型预热完成")
        
    def _monitor_performance(self) -> Dict[str, float]:
        """
        监控系统性能
        
        Returns:
            性能指标字典
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        performance = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_info.percent,
            'available_memory': memory_info.available / (1024**3),  # GB
            'used_memory': memory_info.used / (1024**3)  # GB
        }
        
        return performance
        
    def predict_single_image(self, image_path: str, save_path: str = None, 
                           show_result: bool = False) -> Dict:
        """
        单图片推理
        
        Args:
            image_path: 输入图片路径
            save_path: 结果保存路径
            show_result: 是否显示结果
            
        Returns:
            检测结果字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
                
            LOGGER.info(f"处理图片: {image_path}")
            
            # 开始性能监控
            start_time = time.time()
            perf_before = self._monitor_performance()
            
            # 执行推理
            results = self.model.predict(
                image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device='cpu',
                save=False,
                verbose=False
            )
            
            # 结束性能监控
            end_time = time.time()
            inference_time = end_time - start_time
            perf_after = self._monitor_performance()
            
            # 记录性能数据
            self.inference_times.append(inference_time)
            self.cpu_usage.append(perf_after['cpu_usage'])
            self.memory_usage.append(perf_after['memory_usage'])
            
            # 处理结果
            result_dict = self._process_results(results[0], image_path, inference_time)
            
            # 可视化结果
            if save_path or show_result:
                annotated_image = self._annotate_image(image_path, results[0])
                
                if save_path:
                    cv2.imwrite(save_path, annotated_image)
                    LOGGER.info(f"结果已保存: {save_path}")
                    
                if show_result:
                    cv2.imshow('YOLO26 Drone Detection', annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
            return result_dict
            
        except Exception as e:
            LOGGER.error(f"单图片推理失败: {e}")
            raise
            
    def predict_batch_images(self, image_dir: str, output_dir: str = None, 
                           batch_size: int = 4) -> List[Dict]:
        """
        批量图片推理
        
        Args:
            image_dir: 图片目录路径
            output_dir: 输出目录路径
            batch_size: 批量大小
            
        Returns:
            检测结果列表
        """
        try:
            # 获取图片文件列表
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(image_dir).glob(f'*{ext}'))
                image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
                
            if not image_files:
                raise ValueError(f"在目录 {image_dir} 中未找到图片文件")
                
            LOGGER.info(f"找到 {len(image_files)} 张图片")
            
            # 创建输出目录
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            results_list = []
            total_start_time = time.time()
            
            # 分批处理
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                batch_paths = [str(f) for f in batch_files]
                
                LOGGER.info(f"处理批次 {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}")
                
                # 批量推理
                batch_start_time = time.time()
                perf_before = self._monitor_performance()
                
                batch_results = self.model.predict(
                    batch_paths,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device='cpu',
                    save=False,
                    verbose=False
                )
                
                batch_end_time = time.time()
                batch_inference_time = batch_end_time - batch_start_time
                perf_after = self._monitor_performance()
                
                # 处理批次结果
                for j, (result, image_path) in enumerate(zip(batch_results, batch_paths)):
                    single_inference_time = batch_inference_time / len(batch_results)
                    
                    # 记录性能数据
                    self.inference_times.append(single_inference_time)
                    self.cpu_usage.append(perf_after['cpu_usage'])
                    self.memory_usage.append(perf_after['memory_usage'])
                    
                    # 处理单个结果
                    result_dict = self._process_results(result, image_path, single_inference_time)
                    results_list.append(result_dict)
                    
                    # 保存可视化结果
                    if output_dir:
                        image_name = Path(image_path).stem
                        save_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
                        annotated_image = self._annotate_image(image_path, result)
                        cv2.imwrite(save_path, annotated_image)
                        
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            LOGGER.info(f"批量推理完成 - 总时间: {total_time:.2f}s, 平均每张: {total_time/len(image_files):.3f}s")
            
            return results_list
            
        except Exception as e:
            LOGGER.error(f"批量推理失败: {e}")
            raise
            
    def _process_results(self, result, image_path: str, inference_time: float) -> Dict:
        """
        处理推理结果
        
        Args:
            result: YOLO推理结果
            image_path: 图片路径
            inference_time: 推理时间
            
        Returns:
            处理后的结果字典
        """
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[int(classes[i])] if int(classes[i]) < len(self.class_names) else 'unknown',
                    'confidence': float(confidences[i]),
                    'bbox': {
                        'x1': float(boxes[i][0]),
                        'y1': float(boxes[i][1]),
                        'x2': float(boxes[i][2]),
                        'y2': float(boxes[i][3]),
                        'width': float(boxes[i][2] - boxes[i][0]),
                        'height': float(boxes[i][3] - boxes[i][1])
                    }
                }
                detections.append(detection)
                
        result_dict = {
            'image_path': image_path,
            'inference_time': inference_time,
            'detections_count': len(detections),
            'detections': detections,
            'image_size': {
                'width': result.orig_shape[1],
                'height': result.orig_shape[0]
            }
        }
        
        return result_dict
        
    def _annotate_image(self, image_path: str, result) -> np.ndarray:
        """
        在图片上标注检测结果
        
        Args:
            image_path: 图片路径
            result: YOLO推理结果
            
        Returns:
            标注后的图片
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 创建标注器
        annotator = Annotator(image, line_width=2, font_size=1)
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                # 获取边界框坐标
                x1, y1, x2, y2 = boxes[i]
                confidence = confidences[i]
                class_id = int(classes[i])
                
                # 获取类别名称
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                
                # 创建标签
                label = f'{class_name} {confidence:.2f}'
                
                # 选择颜色
                color = colors(class_id, True)
                
                # 绘制边界框和标签
                annotator.box_label((x1, y1, x2, y2), label, color=color)
                
        return annotator.result()
        
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if not self.inference_times:
            return {}
            
        stats = {
            'total_inferences': len(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'avg_cpu_usage': np.mean(self.cpu_usage),
            'max_cpu_usage': np.max(self.cpu_usage),
            'avg_memory_usage': np.mean(self.memory_usage),
            'max_memory_usage': np.max(self.memory_usage),
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
        
        return stats
        
    def save_performance_report(self, report_path: str):
        """
        保存性能报告
        
        Args:
            report_path: 报告保存路径
        """
        stats = self.get_performance_stats()
        
        report = {
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'performance_stats': stats,
            'detailed_times': self.inference_times,
            'cpu_usage_history': self.cpu_usage,
            'memory_usage_history': self.memory_usage
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        LOGGER.info(f"性能报告已保存: {report_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='YOLO26 CPU优化推理')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--source', type=str, required=True, help='输入源（图片文件或目录）')
    parser.add_argument('--output', type=str, default='', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--save-report', type=str, default='', help='保存性能报告路径')
    
    args = parser.parse_args()
    
    try:
        # 创建推理器
        inferencer = CPUOptimizedInference(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # 处理输入路径
        source_path = args.source
        
        # 如果是相对路径，尝试多种可能的路径组合
        if not os.path.isabs(source_path):
            # 尝试当前目录
            if os.path.exists(source_path):
                source_path = os.path.abspath(source_path)
            # 尝试去掉可能的重复目录前缀
            elif '/' in source_path or '\\' in source_path:
                # 提取文件名
                filename = os.path.basename(source_path)
                if os.path.exists(filename):
                    source_path = os.path.abspath(filename)
                    LOGGER.info(f"路径修正: {args.source} -> {source_path}")
        
        # 判断输入类型
        if os.path.isfile(source_path):
            # 单图片推理
            save_path = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                image_name = Path(source_path).stem
                save_path = os.path.join(args.output, f"{image_name}_detected.jpg")
                
            result = inferencer.predict_single_image(
                image_path=source_path,
                save_path=save_path,
                show_result=args.show
            )
            
            LOGGER.info(f"检测结果: 发现 {result['detections_count']} 个无人机")
            LOGGER.info(f"推理时间: {result['inference_time']:.3f}s")
            
        elif os.path.isdir(source_path):
            # 批量推理
            results = inferencer.predict_batch_images(
                image_dir=source_path,
                output_dir=args.output,
                batch_size=args.batch_size
            )
            
            total_detections = sum(r['detections_count'] for r in results)
            LOGGER.info(f"批量推理完成: 处理 {len(results)} 张图片，发现 {total_detections} 个无人机")
            
        else:
            raise ValueError(f"无效的输入源: {source_path} (原始输入: {args.source})")
            
        # 显示性能统计
        stats = inferencer.get_performance_stats()
        if stats:
            LOGGER.info("=== 性能统计 ===")
            LOGGER.info(f"平均推理时间: {stats['avg_inference_time']:.3f}s")
            LOGGER.info(f"平均FPS: {stats['fps']:.1f}")
            LOGGER.info(f"平均CPU使用率: {stats['avg_cpu_usage']:.1f}%")
            LOGGER.info(f"平均内存使用率: {stats['avg_memory_usage']:.1f}%")
            
        # 保存性能报告
        if args.save_report:
            inferencer.save_performance_report(args.save_report)
            
    except KeyboardInterrupt:
        LOGGER.info("推理被用户中断")
    except Exception as e:
        LOGGER.error(f"推理过程出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()