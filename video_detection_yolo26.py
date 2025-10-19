#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26 Drone Detection 视频流实时检测
支持摄像头、视频文件和RTSP流的实时无人机检测
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors
import psutil
from collections import deque
import queue

class VideoStreamDetector:
    """
    视频流实时检测器
    支持摄像头、视频文件和RTSP流
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, max_fps: int = 30):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            max_fps: 最大帧率限制
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_fps = max_fps
        self.model = None
        self.class_names = ['drone']
        
        # 性能监控
        self.fps_counter = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
        self.frame_times = deque(maxlen=30)
        
        # 线程控制
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # 统计信息
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = None
        
        # CPU优化设置
        self._setup_cpu_optimization()
        
        # 加载模型
        self._load_model()
        
    def _setup_cpu_optimization(self):
        """
        设置CPU优化参数
        """
        num_threads = min(4, os.cpu_count())
        torch.set_num_threads(num_threads)
        
        torch.backends.cudnn.enabled = False
        torch.backends.mkldnn.enabled = True
        
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        
        LOGGER.info(f"CPU优化设置完成 - 线程数: {num_threads}")
        
    def _load_model(self):
        """
        加载YOLO26模型
        """
        try:
            LOGGER.info(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to('cpu')
            
            # 预热模型
            self._warmup_model()
            
            LOGGER.info("模型加载完成")
            
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
            
    def _warmup_model(self, warmup_runs: int = 3):
        """
        预热模型
        
        Args:
            warmup_runs: 预热运行次数
        """
        LOGGER.info("模型预热中...")
        
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
        
    def _create_video_capture(self, source) -> cv2.VideoCapture:
        """
        创建视频捕获对象
        
        Args:
            source: 视频源（摄像头索引、文件路径或RTSP URL）
            
        Returns:
            VideoCapture对象
        """
        try:
            # 判断源类型
            if isinstance(source, int) or source.isdigit():
                # 摄像头
                cap = cv2.VideoCapture(int(source))
                LOGGER.info(f"使用摄像头: {source}")
            elif source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                # 网络流
                cap = cv2.VideoCapture(source)
                LOGGER.info(f"使用网络流: {source}")
            elif os.path.isfile(source):
                # 视频文件
                cap = cv2.VideoCapture(source)
                LOGGER.info(f"使用视频文件: {source}")
            else:
                raise ValueError(f"无效的视频源: {source}")
                
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {source}")
                
            # 设置缓冲区大小
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            LOGGER.info(f"视频信息 - FPS: {fps}, 分辨率: {width}x{height}")
            
            return cap
            
        except Exception as e:
            LOGGER.error(f"创建视频捕获失败: {e}")
            raise
            
    def _detection_worker(self):
        """
        检测工作线程
        """
        while self.running:
            try:
                # 获取帧
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=0.1)
                    frame, frame_id, timestamp = frame_data
                    
                    # 执行检测
                    detection_start = time.time()
                    
                    results = self.model.predict(
                        frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device='cpu',
                        verbose=False
                    )
                    
                    detection_time = time.time() - detection_start
                    
                    # 处理结果
                    detections = self._process_detection_results(results[0])
                    
                    # 发送结果
                    result_data = {
                        'frame': frame,
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'detections': detections,
                        'detection_time': detection_time
                    }
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result_data)
                        
                else:
                    time.sleep(0.001)  # 短暂休眠
                    
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.error(f"检测线程错误: {e}")
                
    def _process_detection_results(self, result) -> List[Dict]:
        """
        处理检测结果
        
        Args:
            result: YOLO检测结果
            
        Returns:
            检测结果列表
        """
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
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
                        'y2': float(boxes[i][3])
                    }
                }
                detections.append(detection)
                
        return detections
        
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict], 
                       fps: float, detection_time: float) -> np.ndarray:
        """
        在帧上标注检测结果和信息
        
        Args:
            frame: 输入帧
            detections: 检测结果
            fps: 当前FPS
            detection_time: 检测耗时
            
        Returns:
            标注后的帧
        """
        annotated_frame = frame.copy()
        
        # 创建标注器
        annotator = Annotator(annotated_frame, line_width=2, font_size=1)
        
        # 标注检测结果
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 创建标签
            label = f'{class_name} {confidence:.2f}'
            
            # 选择颜色
            color = colors(detection['class_id'], True)
            
            # 绘制边界框和标签
            annotator.box_label((x1, y1, x2, y2), label, color=color)
            
        # 添加信息文本
        info_text = [
            f'FPS: {fps:.1f}',
            f'Detection Time: {detection_time*1000:.1f}ms',
            f'Detections: {len(detections)}',
            f'Total Frames: {self.total_frames}',
            f'Total Detections: {self.total_detections}'
        ]
        
        # 绘制信息文本
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            
        return annotated_frame
        
    def detect_video_stream(self, source, output_path: str = None, 
                          display: bool = True, save_detections: bool = False) -> Dict:
        """
        视频流检测主函数
        
        Args:
            source: 视频源
            output_path: 输出视频路径
            display: 是否显示实时画面
            save_detections: 是否保存检测结果
            
        Returns:
            检测统计信息
        """
        try:
            # 创建视频捕获
            cap = self._create_video_capture(source)
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建视频写入器
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                LOGGER.info(f"输出视频: {output_path}")
                
            # 检测结果记录
            detection_log = []
            
            # 启动检测线程
            self.running = True
            detection_thread = threading.Thread(target=self._detection_worker)
            detection_thread.start()
            
            # 初始化统计
            self.start_time = time.time()
            frame_id = 0
            last_fps_time = time.time()
            
            LOGGER.info("开始视频流检测...")
            LOGGER.info("按 'q' 键退出，按 's' 键截图")
            
            try:
                while True:
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        if isinstance(source, str) and os.path.isfile(source):
                            # 视频文件结束
                            LOGGER.info("视频文件播放完毕")
                            break
                        else:
                            # 摄像头或流断开
                            LOGGER.warning("视频流中断，尝试重连...")
                            time.sleep(1)
                            continue
                            
                    frame_start_time = time.time()
                    self.total_frames += 1
                    frame_id += 1
                    
                    # 添加帧到检测队列
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame.copy(), frame_id, frame_start_time))
                        
                    # 获取检测结果
                    annotated_frame = frame.copy()
                    detections = []
                    detection_time = 0
                    
                    if not self.result_queue.empty():
                        try:
                            result_data = self.result_queue.get_nowait()
                            detections = result_data['detections']
                            detection_time = result_data['detection_time']
                            
                            # 更新统计
                            self.total_detections += len(detections)
                            self.detection_history.append(len(detections))
                            
                            # 记录检测结果
                            if save_detections and detections:
                                detection_record = {
                                    'frame_id': frame_id,
                                    'timestamp': frame_start_time,
                                    'detections': detections
                                }
                                detection_log.append(detection_record)
                                
                        except queue.Empty:
                            pass
                            
                    # 计算FPS
                    current_time = time.time()
                    frame_time = current_time - frame_start_time
                    self.frame_times.append(frame_time)
                    
                    if current_time - last_fps_time >= 1.0:
                        if self.frame_times:
                            current_fps = 1.0 / np.mean(self.frame_times)
                            self.fps_counter.append(current_fps)
                        last_fps_time = current_time
                        
                    # 获取当前FPS
                    display_fps = np.mean(self.fps_counter) if self.fps_counter else 0
                    
                    # 标注帧
                    annotated_frame = self._annotate_frame(
                        frame, detections, display_fps, detection_time
                    )
                    
                    # 保存视频
                    if writer:
                        writer.write(annotated_frame)
                        
                    # 显示画面
                    if display:
                        cv2.imshow('YOLO26 Drone Detection - Video Stream', annotated_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            LOGGER.info("用户退出")
                            break
                        elif key == ord('s'):
                            # 截图
                            screenshot_path = f'screenshot_{int(time.time())}.jpg'
                            cv2.imwrite(screenshot_path, annotated_frame)
                            LOGGER.info(f"截图已保存: {screenshot_path}")
                            
                    # FPS限制
                    if self.max_fps > 0:
                        target_frame_time = 1.0 / self.max_fps
                        elapsed = time.time() - frame_start_time
                        if elapsed < target_frame_time:
                            time.sleep(target_frame_time - elapsed)
                            
            except KeyboardInterrupt:
                LOGGER.info("检测被用户中断")
                
            finally:
                # 停止检测线程
                self.running = False
                if detection_thread.is_alive():
                    detection_thread.join(timeout=2)
                    
                # 释放资源
                cap.release()
                if writer:
                    writer.release()
                if display:
                    cv2.destroyAllWindows()
                    
            # 保存检测日志
            if save_detections and detection_log:
                log_path = f'detection_log_{int(time.time())}.json'
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(detection_log, f, indent=2, ensure_ascii=False)
                LOGGER.info(f"检测日志已保存: {log_path}")
                
            # 计算统计信息
            total_time = time.time() - self.start_time
            avg_fps = self.total_frames / total_time if total_time > 0 else 0
            
            stats = {
                'total_frames': self.total_frames,
                'total_detections': self.total_detections,
                'total_time': total_time,
                'average_fps': avg_fps,
                'detection_rate': self.total_detections / self.total_frames if self.total_frames > 0 else 0
            }
            
            LOGGER.info("=== 检测统计 ===")
            LOGGER.info(f"总帧数: {stats['total_frames']}")
            LOGGER.info(f"总检测数: {stats['total_detections']}")
            LOGGER.info(f"总时间: {stats['total_time']:.2f}s")
            LOGGER.info(f"平均FPS: {stats['average_fps']:.2f}")
            LOGGER.info(f"检测率: {stats['detection_rate']:.3f}")
            
            return stats
            
        except Exception as e:
            LOGGER.error(f"视频流检测失败: {e}")
            raise

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='YOLO26视频流实时检测')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--source', type=str, default='0', help='视频源（摄像头索引、文件路径或RTSP URL）')
    parser.add_argument('--output', type=str, default='', help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值')
    parser.add_argument('--max-fps', type=int, default=30, help='最大FPS限制')
    parser.add_argument('--no-display', action='store_true', help='不显示实时画面')
    parser.add_argument('--save-detections', action='store_true', help='保存检测结果日志')
    
    args = parser.parse_args()
    
    try:
        # 创建检测器
        detector = VideoStreamDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_fps=args.max_fps
        )
        
        # 开始检测
        stats = detector.detect_video_stream(
            source=args.source,
            output_path=args.output if args.output else None,
            display=not args.no_display,
            save_detections=args.save_detections
        )
        
    except KeyboardInterrupt:
        LOGGER.info("检测被用户中断")
    except Exception as e:
        LOGGER.error(f"检测过程出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()