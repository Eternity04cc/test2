#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26 OpenVINO优化模块
提供模型转换、优化和加速推理功能
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import deque
import psutil

try:
    import openvino as ov
    from openvino.tools import mo
    from openvino.runtime import Core, Model
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("警告: OpenVINO未安装，请运行: pip install openvino")

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("错误: Ultralytics未安装，请运行: pip install ultralytics")
    sys.exit(1)

class OpenVINOOptimizer:
    """
    OpenVINO模型优化器
    提供YOLO26模型的OpenVINO转换和优化功能
    """
    
    def __init__(self, model_path: str, output_dir: str = "openvino_models"):
        """
        初始化优化器
        
        Args:
            model_path: 原始YOLO模型路径
            output_dir: OpenVINO模型输出目录
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO未安装，无法使用优化功能")
            
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.core = Core()
        self.yolo_model = None
        self.ov_model = None
        self.compiled_model = None
        
        # 模型信息
        self.input_shape = None
        self.output_names = None
        
        LOGGER.info(f"OpenVINO优化器初始化完成")
        LOGGER.info(f"可用设备: {self.core.available_devices}")
        
    def convert_to_openvino(self, precision: str = "FP16", 
                           input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
                           optimize: bool = True) -> str:
        """
        将YOLO模型转换为OpenVINO格式
        
        Args:
            precision: 精度类型 (FP32, FP16, INT8)
            input_shape: 输入形状 (batch, channels, height, width)
            optimize: 是否启用优化
            
        Returns:
            OpenVINO模型文件路径
        """
        try:
            LOGGER.info(f"开始转换模型: {self.model_path}")
            LOGGER.info(f"目标精度: {precision}, 输入形状: {input_shape}")
            
            # 加载YOLO模型
            self.yolo_model = YOLO(self.model_path)
            
            # 导出为ONNX格式
            onnx_path = self.output_dir / "model.onnx"
            LOGGER.info("导出ONNX模型...")
            
            self.yolo_model.export(
                format="onnx",
                imgsz=input_shape[2:],
                optimize=optimize,
                half=(precision == "FP16"),
                simplify=True
            )
            
            # 查找生成的ONNX文件
            model_name = Path(self.model_path).stem
            generated_onnx = Path(self.model_path).parent / f"{model_name}.onnx"
            
            if generated_onnx.exists():
                # 移动到输出目录
                onnx_path = self.output_dir / f"{model_name}.onnx"
                if onnx_path.exists():
                    onnx_path.unlink()
                generated_onnx.rename(onnx_path)
            else:
                raise FileNotFoundError(f"ONNX文件未找到: {generated_onnx}")
                
            LOGGER.info(f"ONNX模型已保存: {onnx_path}")
            
            # 转换为OpenVINO IR格式
            ir_path = self.output_dir / f"{model_name}_{precision.lower()}"
            
            LOGGER.info("转换为OpenVINO IR格式...")
            
            # 使用Model Optimizer转换
            ov_model = mo.convert_model(
                str(onnx_path),
                input_shape=input_shape,
                compress_to_fp16=(precision == "FP16")
            )
            
            # 保存IR模型
            ir_xml_path = ir_path.with_suffix('.xml')
            ov.save_model(ov_model, str(ir_xml_path))
            
            LOGGER.info(f"OpenVINO IR模型已保存: {ir_xml_path}")
            
            # 清理临时ONNX文件
            if onnx_path.exists():
                onnx_path.unlink()
                
            self.input_shape = input_shape
            return str(ir_xml_path)
            
        except Exception as e:
            LOGGER.error(f"模型转换失败: {e}")
            raise
            
    def optimize_model(self, ir_model_path: str, device: str = "CPU") -> None:
        """
        优化OpenVINO模型
        
        Args:
            ir_model_path: IR模型文件路径
            device: 目标设备 (CPU, GPU, AUTO)
        """
        try:
            LOGGER.info(f"优化模型用于设备: {device}")
            
            # 加载模型
            self.ov_model = self.core.read_model(ir_model_path)
            
            # 设置优化配置
            config = {}
            
            if device == "CPU":
                # CPU优化配置
                config = {
                    "PERFORMANCE_HINT": "LATENCY",
                    "CPU_THREADS_NUM": str(min(4, os.cpu_count())),
                    "CPU_BIND_THREAD": "YES",
                    "CPU_THROUGHPUT_STREAMS": "1"
                }
            elif device == "GPU":
                # GPU优化配置
                config = {
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "GPU_THROUGHPUT_STREAMS": "1"
                }
            elif device == "AUTO":
                # 自动设备选择
                config = {
                    "PERFORMANCE_HINT": "LATENCY"
                }
                
            # 编译模型
            self.compiled_model = self.core.compile_model(
                self.ov_model, device, config
            )
            
            # 获取输入输出信息
            self.input_layer = self.compiled_model.input(0)
            self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]
            
            LOGGER.info(f"模型优化完成")
            LOGGER.info(f"输入形状: {self.input_layer.shape}")
            LOGGER.info(f"输出层数: {len(self.output_layers)}")
            
        except Exception as e:
            LOGGER.error(f"模型优化失败: {e}")
            raise
            
    def benchmark_model(self, ir_model_path: str, device: str = "CPU", 
                       num_iterations: int = 100) -> Dict:
        """
        模型性能基准测试
        
        Args:
            ir_model_path: IR模型文件路径
            device: 测试设备
            num_iterations: 测试迭代次数
            
        Returns:
            性能统计信息
        """
        try:
            LOGGER.info(f"开始性能基准测试 - 设备: {device}, 迭代: {num_iterations}")
            
            # 优化模型
            self.optimize_model(ir_model_path, device)
            
            # 准备测试数据
            input_shape = self.input_layer.shape
            test_input = np.random.random(input_shape).astype(np.float32)
            
            # 预热
            LOGGER.info("模型预热中...")
            for _ in range(10):
                _ = self.compiled_model([test_input])
                
            # 性能测试
            LOGGER.info("执行性能测试...")
            inference_times = []
            cpu_usage = []
            memory_usage = []
            
            process = psutil.Process()
            
            for i in range(num_iterations):
                # 记录系统资源
                cpu_percent = psutil.cpu_percent()
                memory_percent = process.memory_percent()
                
                # 推理计时
                start_time = time.perf_counter()
                results = self.compiled_model([test_input])
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                cpu_usage.append(cpu_percent)
                memory_usage.append(memory_percent)
                
                if (i + 1) % 20 == 0:
                    LOGGER.info(f"完成 {i + 1}/{num_iterations} 次推理")
                    
            # 计算统计信息
            stats = {
                'device': device,
                'iterations': num_iterations,
                'input_shape': list(input_shape),
                'inference_times': {
                    'mean': float(np.mean(inference_times)),
                    'std': float(np.std(inference_times)),
                    'min': float(np.min(inference_times)),
                    'max': float(np.max(inference_times)),
                    'median': float(np.median(inference_times)),
                    'p95': float(np.percentile(inference_times, 95)),
                    'p99': float(np.percentile(inference_times, 99))
                },
                'throughput': {
                    'fps': 1000.0 / np.mean(inference_times),
                    'images_per_second': 1000.0 / np.mean(inference_times)
                },
                'system_resources': {
                    'cpu_usage': {
                        'mean': float(np.mean(cpu_usage)),
                        'max': float(np.max(cpu_usage))
                    },
                    'memory_usage': {
                        'mean': float(np.mean(memory_usage)),
                        'max': float(np.max(memory_usage))
                    }
                }
            }
            
            # 输出结果
            LOGGER.info("=== 性能基准测试结果 ===")
            LOGGER.info(f"设备: {stats['device']}")
            LOGGER.info(f"平均推理时间: {stats['inference_times']['mean']:.2f}ms")
            LOGGER.info(f"推理时间标准差: {stats['inference_times']['std']:.2f}ms")
            LOGGER.info(f"最小推理时间: {stats['inference_times']['min']:.2f}ms")
            LOGGER.info(f"最大推理时间: {stats['inference_times']['max']:.2f}ms")
            LOGGER.info(f"P95推理时间: {stats['inference_times']['p95']:.2f}ms")
            LOGGER.info(f"吞吐量: {stats['throughput']['fps']:.2f} FPS")
            LOGGER.info(f"平均CPU使用率: {stats['system_resources']['cpu_usage']['mean']:.1f}%")
            LOGGER.info(f"平均内存使用率: {stats['system_resources']['memory_usage']['mean']:.1f}%")
            
            return stats
            
        except Exception as e:
            LOGGER.error(f"性能基准测试失败: {e}")
            raise

class OpenVINOInference:
    """
    OpenVINO推理引擎
    提供优化的YOLO26推理功能
    """
    
    def __init__(self, ir_model_path: str, device: str = "CPU", 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化推理引擎
        
        Args:
            ir_model_path: OpenVINO IR模型路径
            device: 推理设备
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO未安装，无法使用推理功能")
            
        self.ir_model_path = ir_model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.core = Core()
        self.model = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layers = None
        
        # 类别名称
        self.class_names = ['drone']
        
        # 性能监控
        self.inference_times = deque(maxlen=100)
        
        # 加载和优化模型
        self._load_model()
        
    def _load_model(self):
        """
        加载和编译OpenVINO模型
        """
        try:
            LOGGER.info(f"加载OpenVINO模型: {self.ir_model_path}")
            
            # 读取模型
            self.model = self.core.read_model(self.ir_model_path)
            
            # 设备优化配置
            config = {}
            if self.device == "CPU":
                config = {
                    "PERFORMANCE_HINT": "LATENCY",
                    "CPU_THREADS_NUM": str(min(4, os.cpu_count())),
                    "CPU_BIND_THREAD": "YES"
                }
                
            # 编译模型
            self.compiled_model = self.core.compile_model(
                self.model, self.device, config
            )
            
            # 获取输入输出层
            self.input_layer = self.compiled_model.input(0)
            self.output_layers = [self.compiled_model.output(i) for i in range(len(self.compiled_model.outputs))]
            
            LOGGER.info(f"模型加载完成 - 设备: {self.device}")
            LOGGER.info(f"输入形状: {self.input_layer.shape}")
            
            # 预热模型
            self._warmup_model()
            
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
            
    def _warmup_model(self, warmup_runs: int = 5):
        """
        预热模型
        
        Args:
            warmup_runs: 预热运行次数
        """
        LOGGER.info("OpenVINO模型预热中...")
        
        input_shape = self.input_layer.shape
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        for i in range(warmup_runs):
            try:
                _ = self.compiled_model([dummy_input])
            except Exception as e:
                LOGGER.warning(f"预热运行 {i+1} 失败: {e}")
                
        LOGGER.info("OpenVINO模型预热完成")
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 获取输入尺寸
        input_shape = self.input_layer.shape
        height, width = input_shape[2], input_shape[3]
        
        # 调整大小
        resized = cv2.resize(image, (width, height))
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 添加batch维度
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
        
    def postprocess_outputs(self, outputs: List[np.ndarray], 
                          original_shape: Tuple[int, int]) -> List[Dict]:
        """
        后处理模型输出
        
        Args:
            outputs: 模型输出
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            检测结果列表
        """
        detections = []
        
        # 获取主要输出（通常是第一个）
        output = outputs[0]
        
        # YOLO输出格式: [batch, num_detections, 85] (x, y, w, h, conf, class_probs...)
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度
            
        # 过滤低置信度检测
        confidences = output[:, 4]
        valid_indices = confidences > self.conf_threshold
        valid_detections = output[valid_indices]
        
        if len(valid_detections) == 0:
            return detections
            
        # 提取边界框和置信度
        boxes = valid_detections[:, :4]
        confidences = valid_detections[:, 4]
        class_scores = valid_detections[:, 5:]
        
        # 获取类别ID和分数
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # 最终置信度
        final_confidences = confidences * class_confidences
        
        # 转换边界框格式 (center_x, center_y, width, height) -> (x1, y1, x2, y2)
        input_shape = self.input_layer.shape
        input_height, input_width = input_shape[2], input_shape[3]
        orig_height, orig_width = original_shape
        
        # 缩放因子
        scale_x = orig_width / input_width
        scale_y = orig_height / input_height
        
        x_centers = boxes[:, 0] * scale_x
        y_centers = boxes[:, 1] * scale_y
        widths = boxes[:, 2] * scale_x
        heights = boxes[:, 3] * scale_y
        
        x1s = x_centers - widths / 2
        y1s = y_centers - heights / 2
        x2s = x_centers + widths / 2
        y2s = y_centers + heights / 2
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes=[(float(x1), float(y1), float(w), float(h)) 
                   for x1, y1, w, h in zip(x1s, y1s, widths, heights)],
            scores=final_confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                detection = {
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])] if int(class_ids[i]) < len(self.class_names) else 'unknown',
                    'confidence': float(final_confidences[i]),
                    'bbox': {
                        'x1': float(max(0, x1s[i])),
                        'y1': float(max(0, y1s[i])),
                        'x2': float(min(orig_width, x2s[i])),
                        'y2': float(min(orig_height, y2s[i]))
                    }
                }
                detections.append(detection)
                
        return detections
        
    def predict(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        单张图像推理
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果和推理时间
        """
        try:
            # 记录原始尺寸
            original_shape = image.shape[:2]
            
            # 预处理
            preprocessed = self.preprocess_image(image)
            
            # 推理
            start_time = time.perf_counter()
            outputs = self.compiled_model([preprocessed])
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            self.inference_times.append(inference_time)
            
            # 后处理
            detections = self.postprocess_outputs(
                [output for output in outputs.values()], 
                original_shape
            )
            
            return detections, inference_time
            
        except Exception as e:
            LOGGER.error(f"推理失败: {e}")
            raise
            
    def get_performance_stats(self) -> Dict:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if not self.inference_times:
            return {}
            
        times = list(self.inference_times)
        
        return {
            'count': len(times),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'fps': 1000.0 / np.mean(times)
        }

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='YOLO26 OpenVINO优化工具')
    parser.add_argument('--mode', type=str, choices=['convert', 'benchmark', 'inference'], 
                       required=True, help='运行模式')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output-dir', type=str, default='openvino_models', help='输出目录')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'], 
                       default='FP16', help='模型精度')
    parser.add_argument('--device', type=str, default='CPU', help='推理设备')
    parser.add_argument('--input-shape', type=str, default='1,3,640,640', help='输入形状')
    parser.add_argument('--iterations', type=int, default=100, help='基准测试迭代次数')
    parser.add_argument('--image', type=str, help='测试图像路径')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'convert':
            # 模型转换
            input_shape = tuple(map(int, args.input_shape.split(',')))
            optimizer = OpenVINOOptimizer(args.model, args.output_dir)
            ir_path = optimizer.convert_to_openvino(
                precision=args.precision,
                input_shape=input_shape
            )
            print(f"模型转换完成: {ir_path}")
            
        elif args.mode == 'benchmark':
            # 性能基准测试
            optimizer = OpenVINOOptimizer(args.model, args.output_dir)
            stats = optimizer.benchmark_model(
                args.model, args.device, args.iterations
            )
            
            # 保存结果
            stats_path = Path(args.output_dir) / f"benchmark_{args.device}_{int(time.time())}.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"基准测试结果已保存: {stats_path}")
            
        elif args.mode == 'inference':
            # 推理测试
            if not args.image:
                raise ValueError("推理模式需要指定测试图像")
                
            inference_engine = OpenVINOInference(args.model, args.device)
            
            # 加载图像
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError(f"无法加载图像: {args.image}")
                
            # 执行推理
            detections, inference_time = inference_engine.predict(image)
            
            print(f"推理时间: {inference_time:.2f}ms")
            print(f"检测数量: {len(detections)}")
            
            for i, det in enumerate(detections):
                print(f"检测 {i+1}: {det['class_name']} ({det['confidence']:.3f})")
                
    except Exception as e:
        LOGGER.error(f"执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()