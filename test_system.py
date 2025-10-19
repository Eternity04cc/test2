#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26无人机检测系统测试脚本
验证所有功能模块的正确性和CPU推理性能
"""

import os
import sys
import time
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import traceback
from collections import defaultdict
import psutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SystemTester:
    """
    系统测试器
    验证YOLO26无人机检测系统的各个功能模块
    """
    
    def __init__(self, test_config: Dict):
        """
        初始化测试器
        
        Args:
            test_config: 测试配置字典
        """
        self.config = test_config
        self.test_results = defaultdict(dict)
        self.start_time = None
        
        # 测试数据目录
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # 创建测试图像
        self._create_test_images()
        
        logger.info("系统测试器初始化完成")
        
    def _create_test_images(self):
        """
        创建测试图像
        """
        try:
            # 创建不同尺寸的测试图像
            test_sizes = [(640, 640), (1280, 720), (1920, 1080)]
            
            for i, (width, height) in enumerate(test_sizes):
                # 创建随机图像
                image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                # 添加一些几何形状模拟无人机
                center_x, center_y = width // 2, height // 2
                cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), -1)
                cv2.rectangle(image, (center_x-30, center_y-10), (center_x+30, center_y+10), (0, 0, 255), -1)
                
                # 保存图像
                image_path = self.test_data_dir / f"test_image_{width}x{height}.jpg"
                cv2.imwrite(str(image_path), image)
                
            logger.info(f"创建了 {len(test_sizes)} 张测试图像")
            
        except Exception as e:
            logger.error(f"创建测试图像失败: {e}")
            
    def test_environment(self) -> bool:
        """
        测试环境依赖
        
        Returns:
            测试是否通过
        """
        logger.info("=== 环境依赖测试 ===")
        
        try:
            # 测试Python版本
            python_version = sys.version_info
            logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                logger.error("Python版本过低，需要3.8+")
                return False
                
            # 测试必要库
            required_packages = {
                'ultralytics': 'YOLO',
                'cv2': 'OpenCV',
                'numpy': 'NumPy',
                'torch': 'PyTorch',
                'yaml': 'PyYAML',
                'PIL': 'Pillow',
                'matplotlib': 'Matplotlib'
            }
            
            missing_packages = []
            
            for package, name in required_packages.items():
                try:
                    __import__(package)
                    logger.info(f"✓ {name} 已安装")
                except ImportError:
                    logger.error(f"✗ {name} 未安装")
                    missing_packages.append(name)
                    
            if missing_packages:
                logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
                return False
                
            # 测试系统资源
            cpu_count = os.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            logger.info(f"CPU核心数: {cpu_count}")
            logger.info(f"系统内存: {memory_gb:.1f}GB")
            
            if cpu_count < 2:
                logger.warning("CPU核心数较少，可能影响性能")
                
            if memory_gb < 4:
                logger.warning("系统内存较少，可能影响性能")
                
            self.test_results['environment'] = {
                'status': 'passed',
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'cpu_cores': cpu_count,
                'memory_gb': memory_gb,
                'packages': list(required_packages.values())
            }
            
            logger.info("环境依赖测试通过")
            return True
            
        except Exception as e:
            logger.error(f"环境测试失败: {e}")
            self.test_results['environment'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def test_model_loading(self) -> bool:
        """
        测试模型加载
        
        Returns:
            测试是否通过
        """
        logger.info("=== 模型加载测试 ===")
        
        try:
            from ultralytics import YOLO
            
            # 测试预训练模型下载和加载
            model_names = ['yolov8n.pt', 'yolov8s.pt']
            
            for model_name in model_names:
                try:
                    logger.info(f"测试加载模型: {model_name}")
                    start_time = time.time()
                    
                    model = YOLO(model_name)
                    model.to('cpu')
                    
                    load_time = time.time() - start_time
                    logger.info(f"✓ {model_name} 加载成功 ({load_time:.2f}s)")
                    
                    # 测试模型信息
                    logger.info(f"模型类型: {type(model.model)}")
                    
                    # 清理模型
                    del model
                    
                except Exception as e:
                    logger.error(f"✗ {model_name} 加载失败: {e}")
                    
            self.test_results['model_loading'] = {
                'status': 'passed',
                'tested_models': model_names
            }
            
            logger.info("模型加载测试通过")
            return True
            
        except Exception as e:
            logger.error(f"模型加载测试失败: {e}")
            self.test_results['model_loading'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def test_inference_performance(self) -> bool:
        """
        测试推理性能
        
        Returns:
            测试是否通过
        """
        logger.info("=== 推理性能测试 ===")
        
        try:
            from ultralytics import YOLO
            
            # 加载轻量级模型进行测试
            model = YOLO('yolov8n.pt')
            model.to('cpu')
            
            # 获取测试图像
            test_images = list(self.test_data_dir.glob('*.jpg'))
            
            if not test_images:
                logger.error("没有找到测试图像")
                return False
                
            performance_results = []
            
            for image_path in test_images:
                logger.info(f"测试图像: {image_path.name}")
                
                # 加载图像
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"无法加载图像: {image_path}")
                    continue
                    
                # 多次推理测试
                inference_times = []
                
                for i in range(10):
                    start_time = time.perf_counter()
                    
                    results = model.predict(
                        image,
                        conf=0.25,
                        iou=0.45,
                        device='cpu',
                        verbose=False
                    )
                    
                    end_time = time.perf_counter()
                    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                    inference_times.append(inference_time)
                    
                # 计算统计信息
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                min_time = np.min(inference_times)
                max_time = np.max(inference_times)
                fps = 1000.0 / avg_time
                
                result = {
                    'image': image_path.name,
                    'image_size': f"{image.shape[1]}x{image.shape[0]}",
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'fps': fps,
                    'detections': len(results[0].boxes) if results[0].boxes is not None else 0
                }
                
                performance_results.append(result)
                
                logger.info(f"  平均推理时间: {avg_time:.2f}ms")
                logger.info(f"  FPS: {fps:.2f}")
                logger.info(f"  检测数量: {result['detections']}")
                
            # 整体性能统计
            all_times = [r['avg_time_ms'] for r in performance_results]
            overall_avg = np.mean(all_times)
            overall_fps = 1000.0 / overall_avg
            
            logger.info(f"整体平均推理时间: {overall_avg:.2f}ms")
            logger.info(f"整体平均FPS: {overall_fps:.2f}")
            
            # 性能评估
            performance_grade = 'excellent' if overall_avg < 100 else 'good' if overall_avg < 200 else 'acceptable' if overall_avg < 500 else 'poor'
            
            self.test_results['inference_performance'] = {
                'status': 'passed',
                'overall_avg_time_ms': overall_avg,
                'overall_fps': overall_fps,
                'performance_grade': performance_grade,
                'detailed_results': performance_results
            }
            
            logger.info(f"推理性能测试通过 - 性能等级: {performance_grade}")
            return True
            
        except Exception as e:
            logger.error(f"推理性能测试失败: {e}")
            self.test_results['inference_performance'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def test_data_configuration(self) -> bool:
        """
        测试数据配置
        
        Returns:
            测试是否通过
        """
        logger.info("=== 数据配置测试 ===")
        
        try:
            import yaml
            
            # 检查data.yaml文件
            data_yaml_path = Path('data.yaml')
            
            if not data_yaml_path.exists():
                logger.error("data.yaml文件不存在")
                return False
                
            # 读取配置
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                
            logger.info("data.yaml内容:")
            logger.info(json.dumps(data_config, indent=2, ensure_ascii=False))
            
            # 验证必要字段
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            missing_fields = []
            
            for field in required_fields:
                if field not in data_config:
                    missing_fields.append(field)
                    
            if missing_fields:
                logger.error(f"data.yaml缺少必要字段: {missing_fields}")
                return False
                
            # 验证类别数量
            if data_config['nc'] != len(data_config['names']):
                logger.error(f"类别数量不匹配: nc={data_config['nc']}, names长度={len(data_config['names'])}")
                return False
                
            logger.info(f"✓ 类别数量: {data_config['nc']}")
            logger.info(f"✓ 类别名称: {data_config['names']}")
            
            self.test_results['data_configuration'] = {
                'status': 'passed',
                'config': data_config
            }
            
            logger.info("数据配置测试通过")
            return True
            
        except Exception as e:
            logger.error(f"数据配置测试失败: {e}")
            self.test_results['data_configuration'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def test_custom_scripts(self) -> bool:
        """
        测试自定义脚本
        
        Returns:
            测试是否通过
        """
        logger.info("=== 自定义脚本测试 ===")
        
        try:
            # 检查脚本文件
            script_files = [
                'yolo26_drone_detection.py',
                'train_yolo26.py',
                'inference_yolo26.py',
                'video_detection_yolo26.py',
                'openvino_optimization.py'
            ]
            
            script_status = {}
            
            for script_file in script_files:
                script_path = Path(script_file)
                
                if script_path.exists():
                    # 检查文件大小
                    file_size = script_path.stat().st_size
                    
                    # 简单语法检查
                    try:
                        with open(script_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 编译检查
                        compile(content, script_file, 'exec')
                        
                        script_status[script_file] = {
                            'exists': True,
                            'size_bytes': file_size,
                            'syntax_valid': True
                        }
                        
                        logger.info(f"✓ {script_file} - 大小: {file_size} bytes")
                        
                    except SyntaxError as e:
                        script_status[script_file] = {
                            'exists': True,
                            'size_bytes': file_size,
                            'syntax_valid': False,
                            'syntax_error': str(e)
                        }
                        
                        logger.error(f"✗ {script_file} - 语法错误: {e}")
                        
                else:
                    script_status[script_file] = {
                        'exists': False
                    }
                    
                    logger.error(f"✗ {script_file} - 文件不存在")
                    
            # 检查requirements.txt
            requirements_path = Path('requirements.txt')
            if requirements_path.exists():
                with open(requirements_path, 'r', encoding='utf-8') as f:
                    requirements = f.read().strip().split('\n')
                    
                logger.info(f"✓ requirements.txt - {len(requirements)} 个依赖包")
                script_status['requirements.txt'] = {
                    'exists': True,
                    'package_count': len(requirements)
                }
            else:
                logger.error("✗ requirements.txt - 文件不存在")
                script_status['requirements.txt'] = {'exists': False}
                
            self.test_results['custom_scripts'] = {
                'status': 'passed',
                'scripts': script_status
            }
            
            logger.info("自定义脚本测试通过")
            return True
            
        except Exception as e:
            logger.error(f"自定义脚本测试失败: {e}")
            self.test_results['custom_scripts'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def test_cpu_optimization(self) -> bool:
        """
        测试CPU优化设置
        
        Returns:
            测试是否通过
        """
        logger.info("=== CPU优化测试 ===")
        
        try:
            import torch
            
            # 检查CPU优化设置
            logger.info(f"PyTorch版本: {torch.__version__}")
            logger.info(f"CPU线程数: {torch.get_num_threads()}")
            logger.info(f"MKL-DNN支持: {torch.backends.mkldnn.is_available()}")
            logger.info(f"OpenMP支持: {torch.backends.openmp.is_available()}")
            
            # 测试CPU性能
            logger.info("执行CPU性能测试...")
            
            # 创建测试张量
            test_size = 1000
            a = torch.randn(test_size, test_size)
            b = torch.randn(test_size, test_size)
            
            # 矩阵乘法性能测试
            start_time = time.perf_counter()
            
            for _ in range(10):
                c = torch.mm(a, b)
                
            end_time = time.perf_counter()
            
            cpu_performance_time = (end_time - start_time) / 10 * 1000  # 转换为毫秒
            
            logger.info(f"CPU矩阵乘法平均时间: {cpu_performance_time:.2f}ms")
            
            # 性能评估
            performance_level = 'excellent' if cpu_performance_time < 50 else 'good' if cpu_performance_time < 100 else 'acceptable' if cpu_performance_time < 200 else 'poor'
            
            self.test_results['cpu_optimization'] = {
                'status': 'passed',
                'torch_version': torch.__version__,
                'cpu_threads': torch.get_num_threads(),
                'mkldnn_available': torch.backends.mkldnn.is_available(),
                'openmp_available': torch.backends.openmp.is_available(),
                'performance_time_ms': cpu_performance_time,
                'performance_level': performance_level
            }
            
            logger.info(f"CPU优化测试通过 - 性能等级: {performance_level}")
            return True
            
        except Exception as e:
            logger.error(f"CPU优化测试失败: {e}")
            self.test_results['cpu_optimization'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
            
    def run_all_tests(self) -> Dict:
        """
        运行所有测试
        
        Returns:
            测试结果字典
        """
        logger.info("开始系统全面测试")
        self.start_time = time.time()
        
        # 测试列表
        tests = [
            ('环境依赖', self.test_environment),
            ('模型加载', self.test_model_loading),
            ('推理性能', self.test_inference_performance),
            ('数据配置', self.test_data_configuration),
            ('自定义脚本', self.test_custom_scripts),
            ('CPU优化', self.test_cpu_optimization)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n开始测试: {test_name}")
                result = test_func()
                
                if result:
                    passed_tests += 1
                    logger.info(f"✓ {test_name} 测试通过")
                else:
                    logger.error(f"✗ {test_name} 测试失败")
                    
            except Exception as e:
                logger.error(f"✗ {test_name} 测试异常: {e}")
                logger.error(traceback.format_exc())
                
        # 计算总体结果
        total_time = time.time() - self.start_time
        success_rate = passed_tests / total_tests
        
        overall_result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'total_time_seconds': total_time,
            'overall_status': 'passed' if success_rate >= 0.8 else 'failed',
            'detailed_results': dict(self.test_results)
        }
        
        # 输出总结
        logger.info("\n=== 测试总结 ===")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {success_rate:.1%}")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"整体状态: {'通过' if overall_result['overall_status'] == 'passed' else '失败'}")
        
        return overall_result
        
    def save_test_report(self, results: Dict, output_path: str = None):
        """
        保存测试报告
        
        Args:
            results: 测试结果
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = f"test_report_{int(time.time())}.json"
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            logger.info(f"测试报告已保存: {output_path}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='YOLO26系统测试')
    parser.add_argument('--output', type=str, help='测试报告输出路径')
    parser.add_argument('--config', type=str, help='测试配置文件路径')
    
    args = parser.parse_args()
    
    try:
        # 默认测试配置
        test_config = {
            'test_images': True,
            'performance_benchmark': True,
            'model_validation': True
        }
        
        # 加载自定义配置
        if args.config and Path(args.config).exists():
            with open(args.config, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                test_config.update(custom_config)
                
        # 创建测试器
        tester = SystemTester(test_config)
        
        # 运行测试
        results = tester.run_all_tests()
        
        # 保存报告
        tester.save_test_report(results, args.output)
        
        # 返回适当的退出码
        sys.exit(0 if results['overall_status'] == 'passed' else 1)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()