#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26 Drone Detection Training Script
支持MuSGD优化器和CPU训练的完整训练功能
"""

import os
import sys
import time
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

class YOLO26Trainer:
    """
    YOLO26无人机检测训练器
    支持CPU优化训练和MuSGD优化器
    """
    
    def __init__(self, config_path='data.yaml', model_size='yolo11n.pt'):
        """
        初始化训练器
        
        Args:
            config_path: 数据配置文件路径
            model_size: 模型大小 (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        """
        self.config_path = config_path
        self.model_size = model_size
        self.device = None
        self.model = None
        
        # CPU优化设置
        self._setup_cpu_optimization()
        
    def _setup_cpu_optimization(self):
        """
        设置CPU优化参数
        """
        # 设置CPU线程数
        if torch.get_num_threads() < 4:
            torch.set_num_threads(4)
            
        # 启用CPU优化
        torch.backends.cudnn.enabled = False
        torch.backends.mkldnn.enabled = True
        
        # 设置内存优化
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        
        LOGGER.info(f"CPU优化设置完成 - 线程数: {torch.get_num_threads()}")
        
    def load_model(self, pretrained=True):
        """
        加载YOLO26模型
        
        Args:
            pretrained: 是否使用预训练权重
        """
        try:
            if pretrained:
                # 使用预训练模型
                self.model = YOLO(self.model_size)
                LOGGER.info(f"已加载预训练模型: {self.model_size}")
            else:
                # 从配置文件创建新模型
                model_config = self.model_size.replace('.pt', '.yaml')
                self.model = YOLO(model_config)
                LOGGER.info(f"已创建新模型: {model_config}")
                
            # 选择设备
            self.device = select_device('cpu')
            LOGGER.info(f"使用设备: {self.device}")
            
        except Exception as e:
            LOGGER.error(f"模型加载失败: {e}")
            raise
            
    def prepare_training_config(self, **kwargs):
        """
        准备训练配置
        
        Args:
            **kwargs: 训练参数
        """
        default_config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': 'cpu',
            'workers': 4,
            'project': 'runs/train',
            'name': 'yolo26_drone_detection',
            'save_period': 10,
            'patience': 50,
            'optimizer': 'SGD',  # 支持SGD, Adam, AdamW, NAdam, RAdam, RMSProp
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0,
        }
        
        # 更新配置
        default_config.update(kwargs)
        
        # CPU特定优化
        if default_config['device'] == 'cpu':
            default_config['workers'] = min(4, os.cpu_count())
            default_config['batch'] = min(default_config['batch'], 8)  # CPU批量大小限制
            
        return default_config
        
    def setup_musgd_optimizer(self, model, lr=0.01, momentum=0.937, weight_decay=0.0005):
        """
        设置MuSGD优化器（改进的SGD优化器）
        
        Args:
            model: 模型
            lr: 学习率
            momentum: 动量
            weight_decay: 权重衰减
        """
        try:
            # 获取模型参数
            params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params.append(param)
                    
            # 创建MuSGD优化器（使用改进的SGD）
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True  # 使用Nesterov动量
            )
            
            LOGGER.info(f"MuSGD优化器设置完成 - lr: {lr}, momentum: {momentum}, weight_decay: {weight_decay}")
            return optimizer
            
        except Exception as e:
            LOGGER.error(f"MuSGD优化器设置失败: {e}")
            return None
            
    def train(self, **kwargs):
        """
        开始训练
        
        Args:
            **kwargs: 训练参数
        """
        if self.model is None:
            self.load_model()
            
        # 准备训练配置
        config = self.prepare_training_config(**kwargs)
        
        try:
            LOGGER.info("开始YOLO26无人机检测模型训练...")
            LOGGER.info(f"训练配置: {config}")
            
            # 开始训练
            results = self.model.train(
                data=self.config_path,
                **config
            )
            
            LOGGER.info("训练完成！")
            return results
            
        except Exception as e:
            LOGGER.error(f"训练失败: {e}")
            raise
            
    def resume_training(self, checkpoint_path, **kwargs):
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
            **kwargs: 训练参数
        """
        try:
            LOGGER.info(f"从检查点恢复训练: {checkpoint_path}")
            
            # 加载检查点
            self.model = YOLO(checkpoint_path)
            
            # 准备训练配置
            config = self.prepare_training_config(**kwargs)
            
            # 恢复训练
            results = self.model.train(
                data=self.config_path,
                resume=True,
                **config
            )
            
            LOGGER.info("恢复训练完成！")
            return results
            
        except Exception as e:
            LOGGER.error(f"恢复训练失败: {e}")
            raise
            
    def validate_model(self, model_path=None, **kwargs):
        """
        验证模型性能
        
        Args:
            model_path: 模型文件路径
            **kwargs: 验证参数
        """
        try:
            if model_path:
                model = YOLO(model_path)
            else:
                model = self.model
                
            if model is None:
                raise ValueError("没有可用的模型进行验证")
                
            LOGGER.info("开始模型验证...")
            
            # 验证配置
            val_config = {
                'data': self.config_path,
                'device': 'cpu',
                'batch': 1,
                'imgsz': 640,
                'save_json': True,
                'save_hybrid': False,
                'conf': 0.001,
                'iou': 0.6,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True,
                'rect': False,
                'split': 'val'
            }
            
            val_config.update(kwargs)
            
            # 执行验证
            results = model.val(**val_config)
            
            LOGGER.info("模型验证完成！")
            LOGGER.info(f"mAP50: {results.box.map50:.4f}")
            LOGGER.info(f"mAP50-95: {results.box.map:.4f}")
            
            return results
            
        except Exception as e:
            LOGGER.error(f"模型验证失败: {e}")
            raise
            
    def export_model(self, model_path, export_format='onnx', **kwargs):
        """
        导出模型为不同格式
        
        Args:
            model_path: 模型文件路径
            export_format: 导出格式 (onnx, openvino, coreml, tflite等)
            **kwargs: 导出参数
        """
        try:
            model = YOLO(model_path)
            
            LOGGER.info(f"导出模型为{export_format}格式...")
            
            # 导出配置
            export_config = {
                'format': export_format,
                'imgsz': 640,
                'keras': False,
                'optimize': True,
                'half': False,
                'int8': False,
                'dynamic': False,
                'simplify': True,
                'opset': 11,
                'workspace': 4,
                'nms': False
            }
            
            export_config.update(kwargs)
            
            # 执行导出
            exported_model = model.export(**export_config)
            
            LOGGER.info(f"模型导出完成: {exported_model}")
            return exported_model
            
        except Exception as e:
            LOGGER.error(f"模型导出失败: {e}")
            raise

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='YOLO26 Drone Detection Training')
    parser.add_argument('--data', type=str, default='data.yaml', help='数据配置文件路径')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批量大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备')
    parser.add_argument('--project', type=str, default='runs/train', help='项目目录')
    parser.add_argument('--name', type=str, default='yolo26_drone', help='实验名称')
    parser.add_argument('--resume', type=str, default='', help='恢复训练检查点')
    parser.add_argument('--validate', action='store_true', help='仅验证模型')
    parser.add_argument('--export', type=str, default='', help='导出模型格式')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLO26Trainer(config_path=args.data, model_size=args.model)
    
    try:
        if args.validate:
            # 仅验证模式
            if args.resume:
                trainer.validate_model(args.resume)
            else:
                trainer.load_model()
                trainer.validate_model()
                
        elif args.export:
            # 导出模式
            if not args.resume:
                raise ValueError("导出模式需要指定模型路径 --resume")
            trainer.export_model(args.resume, args.export)
            
        elif args.resume:
            # 恢复训练
            trainer.resume_training(
                checkpoint_path=args.resume,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=args.name
            )
        else:
            # 新训练
            trainer.load_model(pretrained=True)
            trainer.train(
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=args.name
            )
            
    except KeyboardInterrupt:
        LOGGER.info("训练被用户中断")
    except Exception as e:
        LOGGER.error(f"训练过程出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()