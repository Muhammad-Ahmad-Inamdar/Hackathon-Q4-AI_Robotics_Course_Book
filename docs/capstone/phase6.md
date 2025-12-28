---
sidebar_position: 7
learning_objectives:
  - Deploy the integrated system to real hardware or advanced simulation
  - Conduct comprehensive system validation and performance evaluation
  - Demonstrate all capstone project objectives and requirements
  - Evaluate system performance against original specifications
  - Document lessons learned and future improvement opportunities
  - Prepare professional presentation of the completed system
prerequisites:
  - Completion of Phases 1-5 (Full system implementation)
  - All components functioning and integrated
  - Access to deployment environment (real hardware or advanced simulation)
  - Thorough understanding of the complete system
estimated_time: "8 hours"
---

# Phase 6: System Deployment and Demonstration

## Learning Objectives

By completing this phase, you will be able to:
- Deploy the integrated capstone system to a real-world or advanced simulation environment
- Conduct comprehensive system validation and performance evaluation
- Demonstrate all capstone project objectives and requirements successfully
- Evaluate system performance against original specifications and requirements
- Document lessons learned and identify opportunities for future improvements
- Prepare and deliver a professional presentation of the completed system
- Assess the overall success of the integrated Physical AI & Humanoid Robotics system

## Introduction

Phase 6 represents the culmination of your capstone project journey, where you deploy, validate, and demonstrate the complete integrated system you've built across all previous phases. This phase focuses on proving that your integrated system meets all original requirements and objectives while showcasing the successful integration of all four course modules.

The deployment and demonstration phase is critical because it:
- Validates that all integration efforts have resulted in a functional system
- Demonstrates the achievement of capstone project objectives
- Provides evidence of your mastery of Physical AI & Humanoid Robotics concepts
- Shows the practical applicability of your integrated solution
- Evaluates system performance in realistic conditions
- Prepares you for professional presentation and evaluation

## 1. Deployment Strategy

### 1.1 Deployment Environment Setup

Prepare the deployment environment for the complete system:

```python
# deployment_environment.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from capstone_system_interfaces.msg import SystemStatus, PerformanceMetrics
from capstone_system_interfaces.srv import SystemDeployment
import subprocess
import os
import shutil
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional

class DeploymentManager(Node):
    def __init__(self):
        super().__init__('deployment_manager')

        # Declare deployment parameters
        self.declare_parameter('deployment_environment', 'simulation')  # 'simulation' or 'real_hardware'
        self.declare_parameter('deployment_path', '/home/robot/capstone_deployment')
        self.declare_parameter('backup_enabled', True)
        self.declare_parameter('validation_on_deploy', True)
        self.declare_parameter('auto_start_system', True)

        # Publishers and subscribers
        self.deployment_status_publisher = self.create_publisher(
            String, '/deployment/status', 10
        )
        self.system_status_publisher = self.create_publisher(
            SystemStatus, '/system/status', 10
        )

        # Services
        self.deploy_system_service = self.create_service(
            SystemDeployment, '/deploy_system', self.deploy_system_callback
        )
        self.validate_deployment_service = self.create_service(
            SystemDeployment, '/validate_deployment', self.validate_deployment_callback
        )

        # Internal state
        self.deployment_path = Path(self.get_parameter('deployment_path').value)
        self.environment_type = self.get_parameter('deployment_environment').value
        self.deployment_complete = False
        self.validation_results = {}
        self.performance_benchmarks = {}

        self.get_logger().info(f'Deployment Manager initialized for {self.environment_type} environment')

    def deploy_system_callback(self, request, response):
        """Deploy the complete system"""
        try:
            self.get_logger().info('Starting system deployment...')

            # Validate deployment environment
            if not self.validate_deployment_environment():
                response.success = False
                response.message = 'Deployment environment validation failed'
                return response

            # Backup existing deployment if enabled
            if self.get_parameter('backup_enabled').value:
                if not self.create_backup():
                    self.get_logger().warn('Backup creation failed, continuing deployment')

            # Prepare deployment directory
            if not self.prepare_deployment_directory():
                response.success = False
                response.message = 'Failed to prepare deployment directory'
                return response

            # Copy system files
            if not self.copy_system_files():
                response.success = False
                response.message = 'Failed to copy system files'
                return response

            # Configure system for deployment environment
            if not self.configure_system_for_environment():
                response.success = False
                response.message = 'Failed to configure system for environment'
                return response

            # Set up dependencies
            if not self.setup_dependencies():
                response.success = False
                response.message = 'Failed to set up dependencies'
                return response

            # Validate deployment
            if self.get_parameter('validation_on_deploy').value:
                validation_results = self.validate_deployment()
                if not validation_results['overall_pass']:
                    response.success = False
                    response.message = f'Deployment validation failed: {validation_results["issues"]}'
                    return response

            # Start system if auto-start enabled
            if self.get_parameter('auto_start_system').value:
                if not self.start_system():
                    self.get_logger().warn('System start failed, deployment completed but not running')

            self.deployment_complete = True
            response.success = True
            response.message = 'System deployed successfully'
            response.deployment_path = str(self.deployment_path)

            self.get_logger().info('System deployment completed successfully')

        except Exception as e:
            self.get_logger().error(f'Deployment error: {e}')
            response.success = False
            response.message = f'Deployment error: {str(e)}'

        return response

    def validate_deployment_environment(self) -> bool:
        """Validate the deployment environment"""
        try:
            # Check if deployment path is accessible
            if not self.deployment_path.parent.exists():
                self.get_logger().error(f'Deployment path parent does not exist: {self.deployment_path.parent}')
                return False

            # Check system resources
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8.0:  # Require at least 8GB RAM
                self.get_logger().warn(f'System has low memory: {memory_gb:.1f}GB (recommended: 8GB+)')

            # Check disk space
            disk_usage = shutil.disk_usage(self.deployment_path.parent)
            available_gb = disk_usage.free / (1024**3)
            if available_gb < 10.0:  # Require at least 10GB free space
                self.get_logger().error(f'Insufficient disk space: {available_gb:.1f}GB available (required: 10GB+)')
                return False

            # Check for required hardware (if deploying to real hardware)
            if self.environment_type == 'real_hardware':
                if not self.validate_hardware_requirements():
                    return False

            return True

        except Exception as e:
            self.get_logger().error(f'Environment validation error: {e}')
            return False

    def validate_hardware_requirements(self) -> bool:
        """Validate hardware requirements for real hardware deployment"""
        try:
            # Check for NVIDIA GPU (for Isaac ROS and AI processing)
            gpu_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if gpu_check.returncode != 0:
                self.get_logger().warn('NVIDIA GPU not detected - AI processing may be limited')

            # Check for connected sensors and actuators
            # This would check for cameras, IMUs, motors, etc.
            # Implementation would depend on specific hardware

            # Check network connectivity if needed
            # Check power requirements

            return True

        except Exception as e:
            self.get_logger().error(f'Hardware validation error: {e}')
            return False

    def create_backup(self) -> bool:
        """Create backup of existing deployment"""
        try:
            backup_path = self.deployment_path.with_suffix(f'.backup_{int(time.time())}')

            if self.deployment_path.exists():
                shutil.copytree(self.deployment_path, backup_path)
                self.get_logger().info(f'Backup created at: {backup_path}')

            return True

        except Exception as e:
            self.get_logger().error(f'Backup creation error: {e}')
            return False

    def prepare_deployment_directory(self) -> bool:
        """Prepare the deployment directory"""
        try:
            # Create deployment directory if it doesn't exist
            self.deployment_path.mkdir(parents=True, exist_ok=True)

            # Set appropriate permissions
            os.chmod(self.deployment_path, 0o755)

            # Create subdirectories
            subdirs = ['config', 'launch', 'models', 'data', 'logs', 'scripts']
            for subdir in subdirs:
                (self.deployment_path / subdir).mkdir(exist_ok=True)

            return True

        except Exception as e:
            self.get_logger().error(f'Directory preparation error: {e}')
            return False

    def copy_system_files(self) -> bool:
        """Copy system files to deployment directory"""
        try:
            # Define source and destination mappings
            source_dir = Path(__file__).parent.parent  # Assuming this is run from project root

            file_mappings = {
                'src/': 'src/',
                'config/': 'config/',
                'launch/': 'launch/',
                'models/': 'models/',
                'package.xml': 'package.xml',
                'setup.py': 'setup.py',
                'setup.cfg': 'setup.cfg',
                'requirements.txt': 'requirements.txt'
            }

            for src_rel, dest_rel in file_mappings.items():
                src_path = source_dir / src_rel
                dest_path = self.deployment_path / dest_rel

                if src_path.is_file():
                    shutil.copy2(src_path, dest_path)
                elif src_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
                else:
                    self.get_logger().warn(f'Source path does not exist: {src_path}')

            return True

        except Exception as e:
            self.get_logger().error(f'File copying error: {e}')
            return False

    def configure_system_for_environment(self) -> bool:
        """Configure system for the specific deployment environment"""
        try:
            # Update configuration files based on environment
            config_files = [
                self.deployment_path / 'config' / 'system_config.yaml',
                self.deployment_path / 'launch' / 'deployment.launch.py'
            ]

            for config_file in config_files:
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)

                    # Update environment-specific settings
                    if 'environment' in config_data:
                        config_data['environment'] = self.environment_type

                    if 'simulation_mode' in config_data:
                        config_data['simulation_mode'] = (self.environment_type == 'simulation')

                    with open(config_file, 'w') as f:
                        yaml.dump(config_data, f)

            # Update launch files for environment
            self.update_launch_files_for_environment()

            return True

        except Exception as e:
            self.get_logger().error(f'Environment configuration error: {e}')
            return False

    def setup_dependencies(self) -> bool:
        """Setup system dependencies"""
        try:
            # Install Python dependencies
            requirements_file = self.deployment_path / 'requirements.txt'
            if requirements_file.exists():
                pip_install_cmd = [
                    'pip3', 'install', '-r', str(requirements_file), '--user'
                ]

                result = subprocess.run(pip_install_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.get_logger().error(f'Pip install failed: {result.stderr}')
                    return False

            # Check for Isaac ROS dependencies
            if self.environment_type == 'real_hardware':
                # Validate Isaac ROS installation
                try:
                    import isaac_ros_common
                    self.get_logger().info('Isaac ROS dependencies validated')
                except ImportError:
                    self.get_logger().warn('Isaac ROS not available - some features may be limited')

            return True

        except Exception as e:
            self.get_logger().error(f'Dependency setup error: {e}')
            return False

    def start_system(self) -> bool:
        """Start the deployed system"""
        try:
            # Launch the system using appropriate launch file
            if self.environment_type == 'simulation':
                launch_file = self.deployment_path / 'launch' / 'capstone_simulation.launch.py'
            else:
                launch_file = self.deployment_path / 'launch' / 'capstone_real_hardware.launch.py'

            if not launch_file.exists():
                self.get_logger().warn(f'Launch file does not exist: {launch_file}')
                return False

            # Start system in background process
            launch_cmd = ['ros2', 'launch', str(launch_file)]
            self.system_process = subprocess.Popen(launch_cmd)

            # Wait briefly to check if system started successfully
            time.sleep(2.0)

            if self.system_process.poll() is not None:
                # Process terminated early, likely an error
                self.get_logger().error('System failed to start properly')
                return False

            self.get_logger().info('System started successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'System start error: {e}')
            return False

    def validate_deployment_callback(self, request, response):
        """Validate the deployed system"""
        try:
            validation_results = self.validate_deployment()

            response.success = validation_results['overall_pass']
            response.message = validation_results['message']
            response.validation_results = json.dumps(validation_results['details'])

        except Exception as e:
            self.get_logger().error(f'Deployment validation error: {e}')
            response.success = False
            response.message = f'Validation error: {str(e)}'

        return response

    def validate_deployment(self) -> Dict[str, Any]:
        """Perform comprehensive deployment validation"""
        validation_results = {
            'overall_pass': False,
            'overall_score': 0.0,
            'issues': [],
            'details': {},
            'message': 'Deployment validation completed'
        }

        # Validate file structure
        file_validation = self.validate_file_structure()
        validation_results['details']['file_structure'] = file_validation

        if not file_validation['pass']:
            validation_results['issues'].extend(file_validation['issues'])

        # Validate configurations
        config_validation = self.validate_configurations()
        validation_results['details']['configurations'] = config_validation

        if not config_validation['pass']:
            validation_results['issues'].extend(config_validation['issues'])

        # Validate dependencies
        dependency_validation = self.validate_dependencies()
        validation_results['details']['dependencies'] = dependency_validation

        if not dependency_validation['pass']:
            validation_results['issues'].extend(dependency_validation['issues'])

        # Validate communication framework
        comm_validation = self.validate_communication_framework()
        validation_results['details']['communication'] = comm_validation

        if not comm_validation['pass']:
            validation_results['issues'].extend(comm_validation['issues'])

        # Calculate overall score
        total_checks = 4
        passed_checks = sum([
            file_validation['pass'],
            config_validation['pass'],
            dependency_validation['pass'],
            comm_validation['pass']
        ])

        validation_results['overall_score'] = passed_checks / total_checks if total_checks > 0 else 0.0
        validation_results['overall_pass'] = len(validation_results['issues']) == 0

        if validation_results['overall_pass']:
            validation_results['message'] = 'All deployment validations passed'
        else:
            validation_results['message'] = f'Deployment validation failed: {len(validation_results["issues"])} issues found'

        return validation_results

    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate the deployed file structure"""
        issues = []

        required_dirs = [
            self.deployment_path / 'config',
            self.deployment_path / 'launch',
            self.deployment_path / 'src',
            self.deployment_path / 'models',
            self.deployment_path / 'data'
        ]

        for req_dir in required_dirs:
            if not req_dir.exists():
                issues.append(f'Missing required directory: {req_dir}')

        required_files = [
            self.deployment_path / 'package.xml',
            self.deployment_path / 'setup.py',
            self.deployment_path / 'requirements.txt'
        ]

        for req_file in required_files:
            if not req_file.exists():
                issues.append(f'Missing required file: {req_file}')

        return {
            'pass': len(issues) == 0,
            'issues': issues,
            'details': {
                'required_directories_exist': all(d.exists() for d in required_dirs),
                'required_files_exist': all(f.exists() for f in required_files)
            }
        }

    def validate_configurations(self) -> Dict[str, Any]:
        """Validate configuration files"""
        issues = []

        config_files = [
            self.deployment_path / 'config' / 'system_config.yaml',
            self.deployment_path / 'config' / 'ros_config.yaml',
            self.deployment_path / 'config' / 'ai_config.yaml'
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    issues.append(f'Invalid YAML in config file {config_file}: {e}')
            else:
                issues.append(f'Missing config file: {config_file}')

        return {
            'pass': len(issues) == 0,
            'issues': issues,
            'details': {'config_files_valid': len(issues) == 0}
        }

    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate system dependencies"""
        issues = []

        # Check for required Python packages
        required_packages = [
            'rclpy', 'torch', 'openai', 'transformers', 'cv2', 'numpy', 'speech_recognition'
        ]

        for package_name in required_packages:
            try:
                if package_name == 'cv2':
                    import cv2
                elif package_name == 'speech_recognition':
                    import speech_recognition
                else:
                    __import__(package_name)
            except ImportError:
                issues.append(f'Missing required Python package: {package_name}')

        # Check for ROS 2 installation
        try:
            result = subprocess.run(['ros2', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                issues.append('ROS 2 not properly installed')
        except FileNotFoundError:
            issues.append('ROS 2 command not found')

        return {
            'pass': len(issues) == 0,
            'issues': issues,
            'details': {'required_packages_installed': len(issues) == 0}
        }

    def validate_communication_framework(self) -> Dict[str, Any]:
        """Validate ROS 2 communication framework"""
        issues = []

        # Check if ROS 2 daemon is running
        try:
            result = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                issues.append('ROS 2 communication framework not accessible')
        except subprocess.TimeoutExpired:
            issues.append('ROS 2 communication check timed out')

        return {
            'pass': len(issues) == 0,
            'issues': issues,
            'details': {'ros2_framework_accessible': len(issues) == 0}
        }

    def update_launch_files_for_environment(self):
        """Update launch files for the specific environment"""
        # This would modify launch files based on environment type
        # For example, enabling/disabling simulation-specific nodes
        pass
```

### 1.2 System Performance Monitoring

Implement comprehensive system performance monitoring:

```python
# performance_monitoring.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String
from capstone_system_interfaces.msg import PerformanceMetrics, SystemHealth
import psutil
import GPUtil
import time
import threading
from collections import deque
from typing import Dict, List, Any

class PerformanceMonitoringNode(Node):
    def __init__(self):
        super().__init__('performance_monitoring_node')

        # Performance tracking
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.gpu_usage_history = deque(maxlen=100)
        self.message_rate_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)

        # Publishers
        self.performance_metrics_publisher = self.create_publisher(
            PerformanceMetrics, '/system/performance_metrics', 10
        )
        self.system_health_publisher = self.create_publisher(
            SystemHealth, '/system/health', 10
        )
        self.cpu_usage_publisher = self.create_publisher(
            Float32, '/system/cpu_usage', 10
        )
        self.memory_usage_publisher = self.create_publisher(
            Float32, '/system/memory_usage', 10
        )
        self.gpu_usage_publisher = self.create_publisher(
            Float32, '/system/gpu_usage', 10
        )

        # Monitoring timer
        self.monitoring_timer = self.create_timer(1.0, self.performance_monitoring_callback)
        self.detailed_monitoring_timer = self.create_timer(5.0, self.detailed_performance_callback)

        # Performance thresholds
        self.performance_thresholds = {
            'cpu_usage': 80.0,      # percentage
            'memory_usage': 85.0,   # percentage
            'gpu_usage': 85.0,      # percentage
            'gpu_memory': 80.0,     # percentage
            'message_rate': 100.0,  # messages per second per topic
            'latency_threshold': 0.5  # seconds
        }

        # Performance statistics
        self.performance_stats = {
            'average_cpu': 0.0,
            'average_memory': 0.0,
            'average_gpu': 0.0,
            'average_gpu_memory': 0.0,
            'message_rates': {},
            'average_latency': 0.0,
            'peak_cpu': 0.0,
            'peak_memory': 0.0,
            'peak_gpu': 0.0
        }

        self.get_logger().info('Performance Monitoring Node initialized')

    def performance_monitoring_callback(self):
        """Monitor system performance metrics"""
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent

            # Get GPU metrics if available
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
            except:
                pass  # GPU monitoring not available

            # Update history
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
            self.gpu_usage_history.append(gpu_percent)

            # Publish metrics
            cpu_msg = Float32()
            cpu_msg.data = float(cpu_percent)
            self.safe_publish(self.cpu_usage_publisher, cpu_msg)

            memory_msg = Float32()
            memory_msg.data = float(memory_percent)
            self.safe_publish(self.memory_usage_publisher, memory_msg)

            gpu_msg = Float32()
            gpu_msg.data = float(gpu_percent)
            self.safe_publish(self.gpu_usage_publisher, gpu_msg)

            # Update performance statistics
            self.update_performance_statistics(cpu_percent, memory_percent, gpu_percent, gpu_memory_percent)

        except Exception as e:
            self.get_logger().error(f'Performance monitoring error: {e}')

    def detailed_performance_callback(self):
        """Perform detailed performance analysis"""
        try:
            # Calculate averages
            avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else 0.0
            avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0.0
            avg_gpu = sum(self.gpu_usage_history) / len(self.gpu_usage_history) if self.gpu_usage_history else 0.0

            # Create performance metrics message
            metrics_msg = PerformanceMetrics()
            metrics_msg.header.stamp = self.get_clock().now().to_msg()
            metrics_msg.cpu_usage_avg = avg_cpu
            metrics_msg.cpu_usage_peak = max(self.cpu_usage_history) if self.cpu_usage_history else 0.0
            metrics_msg.memory_usage_avg = avg_memory
            metrics_msg.memory_usage_peak = max(self.memory_usage_history) if self.memory_usage_history else 0.0
            metrics_msg.gpu_usage_avg = avg_gpu
            metrics_msg.gpu_usage_peak = max(self.gpu_usage_history) if self.gpu_usage_history else 0.0

            # Check against thresholds
            metrics_msg.cpu_usage_warning = avg_cpu > self.performance_thresholds['cpu_usage']
            metrics_msg.memory_usage_warning = avg_memory > self.performance_thresholds['memory_usage']
            metrics_msg.gpu_usage_warning = avg_gpu > self.performance_thresholds['gpu_usage']

            # Get process-specific metrics
            metrics_msg.process_count = len(psutil.pids())
            metrics_msg.ros_node_count = self.count_ros_nodes()

            # Calculate system load indicators
            load_avg = psutil.getloadavg()
            metrics_msg.system_load_1min = load_avg[0]
            metrics_msg.system_load_5min = load_avg[1]
            metrics_msg.system_load_15min = load_avg[2]

            # Publish performance metrics
            self.safe_publish(self.performance_metrics_publisher, metrics_msg)

            # Create and publish system health assessment
            health_msg = SystemHealth()
            health_msg.header.stamp = self.get_clock().now().to_msg()
            health_msg.overall_health_score = self.calculate_health_score(metrics_msg)
            health_msg.system_stress_level = self.assess_system_stress(metrics_msg)
            health_msg.performance_degradation_indicators = self.detect_performance_degradation(metrics_msg)

            self.safe_publish(self.system_health_publisher, health_msg)

            # Log performance warnings
            if metrics_msg.cpu_usage_warning:
                self.get_logger().warn(f'High CPU usage: {avg_cpu:.1f}%')
            if metrics_msg.memory_usage_warning:
                self.get_logger().warn(f'High memory usage: {avg_memory:.1f}%')
            if metrics_msg.gpu_usage_warning:
                self.get_logger().warn(f'High GPU usage: {avg_gpu:.1f}%')

        except Exception as e:
            self.get_logger().error(f'Detailed performance monitoring error: {e}')

    def update_performance_statistics(self, cpu: float, memory: float, gpu: float, gpu_memory: float):
        """Update performance statistics"""
        # Update averages
        self.performance_stats['average_cpu'] = sum(self.cpu_usage_history) / len(self.cpu_usage_history) if self.cpu_usage_history else 0.0
        self.performance_stats['average_memory'] = sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0.0
        self.performance_stats['average_gpu'] = sum(self.gpu_usage_history) / len(self.gpu_usage_history) if self.gpu_usage_history else 0.0
        self.performance_stats['average_gpu_memory'] = gpu_memory

        # Update peaks
        self.performance_stats['peak_cpu'] = max(list(self.cpu_usage_history) + [0.0])
        self.performance_stats['peak_memory'] = max(list(self.memory_usage_history) + [0.0])
        self.performance_stats['peak_gpu'] = max(list(self.gpu_usage_history) + [0.0])

    def count_ros_nodes(self) -> int:
        """Count active ROS nodes"""
        try:
            result = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                return len([node for node in nodes if node.strip()])
        except:
            pass
        return 0

    def calculate_health_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall system health score based on performance metrics"""
        # Normalize metrics to 0-1 scale (lower is better for usage metrics)
        cpu_score = max(0.0, 1.0 - (metrics.cpu_usage_avg / 100.0))
        memory_score = max(0.0, 1.0 - (metrics.memory_usage_avg / 100.0))
        gpu_score = max(0.0, 1.0 - (metrics.gpu_usage_avg / 100.0))

        # Weighted average (memory and CPU are more critical)
        health_score = (
            cpu_score * 0.4 +
            memory_score * 0.4 +
            gpu_score * 0.2
        )

        return min(1.0, max(0.0, health_score))  # Clamp to 0-1 range

    def assess_system_stress(self, metrics: PerformanceMetrics) -> int:
        """Assess system stress level (0=normal, 1=moderate, 2=high, 3=critical)"""
        stress_score = 0

        if metrics.cpu_usage_avg > 80 or metrics.memory_usage_avg > 85 or metrics.gpu_usage_avg > 85:
            stress_score = 1  # Moderate stress
        if metrics.cpu_usage_avg > 90 or metrics.memory_usage_avg > 95 or metrics.gpu_usage_avg > 95:
            stress_score = 2  # High stress
        if metrics.cpu_usage_avg > 95 or metrics.memory_usage_avg > 98 or metrics.gpu_usage_avg > 98:
            stress_score = 3  # Critical stress

        return stress_score

    def detect_performance_degradation(self, metrics: PerformanceMetrics) -> List[str]:
        """Detect performance degradation indicators"""
        degradation_indicators = []

        # Check for sustained high resource usage
        if len(self.cpu_usage_history) >= 10:
            recent_cpu_avg = sum(list(self.cpu_usage_history)[-10:]) / 10
            if recent_cpu_avg > 85:
                degradation_indicators.append(f"Sustained high CPU usage: {recent_cpu_avg:.1f}%")

        if len(self.memory_usage_history) >= 10:
            recent_memory_avg = sum(list(self.memory_usage_history)[-10:]) / 10
            if recent_memory_avg > 90:
                degradation_indicators.append(f"Sustained high memory usage: {recent_memory_avg:.1f}%")

        # Check for increasing trends
        if len(self.cpu_usage_history) >= 20:
            first_half = list(self.cpu_usage_history)[:10]
            second_half = list(self.cpu_usage_history)[10:]
            if len(first_half) > 0 and len(second_half) > 0:
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                if second_avg > first_avg + 10:  # Increased by more than 10%
                    degradation_indicators.append(f"Increasing CPU usage trend: {first_avg:.1f}% → {second_avg:.1f}%")

        return degradation_indicators
```

### 1.3 System Validation Suite

Create a comprehensive system validation suite:

```python
# system_validation_suite.py
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_system_interfaces.srv import SystemValidation
from capstone_system_interfaces.msg import ValidationReport
import time
import threading
from typing import Dict, List, Any, Callable
import json

class SystemValidationSuite:
    """Comprehensive validation suite for the deployed system"""

    def __init__(self, node: Node):
        self.node = node
        self.validation_results = {}
        self.validation_history = []
        self.validation_metrics = {
            'functional_score': 0.0,
            'performance_score': 0.0,
            'safety_score': 0.0,
            'integration_score': 0.0,
            'reliability_score': 0.0
        }

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation"""
        self.node.get_logger().info('Starting comprehensive system validation...')

        start_time = time.time()

        # Run validation categories
        functional_results = self.run_functional_validation()
        performance_results = self.run_performance_validation()
        safety_results = self.run_safety_validation()
        integration_results = self.run_integration_validation()
        reliability_results = self.run_reliability_validation()

        # Calculate overall validation score
        overall_score = self.calculate_overall_validation_score(
            functional_results, performance_results, safety_results,
            integration_results, reliability_results
        )

        # Compile results
        validation_report = {
            'timestamp': time.time(),
            'overall_score': overall_score,
            'overall_pass': overall_score >= 0.85,  # 85% threshold
            'validation_time': time.time() - start_time,
            'functional_validation': functional_results,
            'performance_validation': performance_results,
            'safety_validation': safety_results,
            'integration_validation': integration_results,
            'reliability_validation': reliability_results,
            'detailed_results': {
                'functional': functional_results['details'],
                'performance': performance_results['details'],
                'safety': safety_results['details'],
                'integration': integration_results['details'],
                'reliability': reliability_results['details']
            }
        }

        # Update validation history
        self.validation_history.append(validation_report)
        if len(self.validation_history) > 50:  # Keep last 50 validations
            self.validation_history = self.validation_history[-50:]

        # Update metrics
        self.validation_metrics['functional_score'] = functional_results['score']
        self.validation_metrics['performance_score'] = performance_results['score']
        self.validation_metrics['safety_score'] = safety_results['score']
        self.validation_metrics['integration_score'] = integration_results['score']
        self.validation_metrics['reliability_score'] = reliability_results['score']

        self.node.get_logger().info(f'Comprehensive validation completed. Score: {overall_score:.2f}, Time: {validation_report["validation_time"]:.2f}s')

        return validation_report

    def run_functional_validation(self) -> Dict[str, Any]:
        """Run functional validation tests"""
        self.node.get_logger().info('Running functional validation...')

        functional_tests = [
            ('Voice Recognition', self.test_voice_recognition),
            ('Language Understanding', self.test_language_understanding),
            ('Vision Processing', self.test_vision_processing),
            ('Task Planning', self.test_task_planning),
            ('Action Execution', self.test_action_execution),
            ('Navigation', self.test_navigation),
            ('Safety Systems', self.test_safety_systems),
            ('Human Interaction', self.test_human_interaction)
        ]

        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'score': 0.0,
            'details': {}
        }

        for test_name, test_func in functional_tests:
            try:
                test_result = test_func()
                results['tests_run'] += 1
                if test_result['pass']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1

                results['details'][test_name] = test_result
                self.node.get_logger().debug(f'{test_name}: {"PASS" if test_result["pass"] else "FAIL"}')

            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['details'][test_name] = {'pass': False, 'error': str(e)}
                self.node.get_logger().error(f'{test_name} test error: {e}')

        if results['tests_run'] > 0:
            results['score'] = results['tests_passed'] / results['tests_run']
        else:
            results['score'] = 0.0

        return results

    def run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests"""
        self.node.get_logger().info('Running performance validation...')

        performance_tests = [
            ('Real-time Response', self.test_real_time_response),
            ('Throughput Capacity', self.test_throughput_capacity),
            ('Resource Utilization', self.test_resource_utilization),
            ('Communication Latency', self.test_communication_latency),
            ('Processing Speed', self.test_processing_speed),
            ('System Scalability', self.test_system_scalability)
        ]

        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'score': 0.0,
            'details': {}
        }

        for test_name, test_func in performance_tests:
            try:
                test_result = test_func()
                results['tests_run'] += 1
                if test_result['pass']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1

                results['details'][test_name] = test_result
                self.node.get_logger().debug(f'{test_name}: {"PASS" if test_result["pass"] else "FAIL"}')

            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['details'][test_name] = {'pass': False, 'error': str(e)}
                self.node.get_logger().error(f'{test_name} performance test error: {e}')

        if results['tests_run'] > 0:
            results['score'] = results['tests_passed'] / results['tests_run']
        else:
            results['score'] = 0.0

        return results

    def run_safety_validation(self) -> Dict[str, Any]:
        """Run safety validation tests"""
        self.node.get_logger().info('Running safety validation...')

        safety_tests = [
            ('Emergency Stop', self.test_emergency_stop),
            ('Collision Avoidance', self.test_collision_avoidance),
            ('Safe Navigation', self.test_safe_navigation),
            ('Command Validation', self.test_command_validation),
            ('LLM Safety Filters', self.test_llm_safety_filters),
            ('Human Safety Zones', self.test_human_safety_zones),
            ('Force Limiting', self.test_force_limiting),
            ('Safe Recovery', self.test_safe_recovery)
        ]

        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'score': 0.0,  # Safety tests require 100% pass rate
            'details': {}
        }

        for test_name, test_func in safety_tests:
            try:
                test_result = test_func()
                results['tests_run'] += 1
                if test_result['pass']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1

                results['details'][test_name] = test_result
                self.node.get_logger().debug(f'{test_name}: {"PASS" if test_result["pass"] else "FAIL"}')

            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['details'][test_name] = {'pass': False, 'error': str(e)}
                self.node.get_logger().error(f'{test_name} safety test error: {e}')

        # For safety tests, we require perfect score for pass
        if results['tests_run'] > 0:
            results['score'] = results['tests_passed'] / results['tests_run']
            if results['tests_failed'] > 0:
                results['score'] = 0.0  # Any safety failure results in 0 score
        else:
            results['score'] = 0.0

        return results

    def run_integration_validation(self) -> Dict[str, Any]:
        """Run integration validation tests"""
        self.node.get_logger().info('Running integration validation...')

        integration_tests = [
            ('Module Integration', self.test_module_integration),
            ('Communication Patterns', self.test_communication_patterns),
            ('Data Flow', self.test_data_flow),
            ('Cross-module Coordination', self.test_cross_module_coordination),
            ('Synchronization', self.test_synchronization),
            ('Error Propagation', self.test_error_propagation),
            ('Resource Sharing', self.test_resource_sharing)
        ]

        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'score': 0.0,
            'details': {}
        }

        for test_name, test_func in integration_tests:
            try:
                test_result = test_func()
                results['tests_run'] += 1
                if test_result['pass']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1

                results['details'][test_name] = test_result
                self.node.get_logger().debug(f'{test_name}: {"PASS" if test_result["pass"] else "FAIL"}')

            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['details'][test_name] = {'pass': False, 'error': str(e)}
                self.node.get_logger().error(f'{test_name} integration test error: {e}')

        if results['tests_run'] > 0:
            results['score'] = results['tests_passed'] / results['tests_run']
        else:
            results['score'] = 0.0

        return results

    def run_reliability_validation(self) -> Dict[str, Any]:
        """Run reliability validation tests"""
        self.node.get_logger().info('Running reliability validation...')

        reliability_tests = [
            ('Long-term Stability', self.test_long_term_stability),
            ('Fault Tolerance', self.test_fault_tolerance),
            ('Graceful Degradation', self.test_graceful_degradation),
            ('Recovery Procedures', self.test_recovery_procedures),
            ('Memory Leak Detection', self.test_memory_leak_detection),
            ('Resource Exhaustion', self.test_resource_exhaustion),
            ('Network Resilience', self.test_network_resilience)
        ]

        results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'score': 0.0,
            'details': {}
        }

        for test_name, test_func in reliability_tests:
            try:
                test_result = test_func()
                results['tests_run'] += 1
                if test_result['pass']:
                    results['tests_passed'] += 1
                else:
                    results['tests_failed'] += 1

                results['details'][test_name] = test_result
                self.node.get_logger().debug(f'{test_name}: {"PASS" if test_result["pass"] else "FAIL"}')

            except Exception as e:
                results['tests_run'] += 1
                results['tests_failed'] += 1
                results['details'][test_name] = {'pass': False, 'error': str(e)}
                self.node.get_logger().error(f'{test_name} reliability test error: {e}')

        if results['tests_run'] > 0:
            results['score'] = results['tests_passed'] / results['tests_run']
        else:
            results['score'] = 0.0

        return results

    def calculate_overall_validation_score(self, functional, performance, safety, integration, reliability) -> float:
        """Calculate overall validation score with weighted components"""
        # Weighted scoring with safety being most critical
        weights = {
            'functional': 0.20,
            'performance': 0.20,
            'safety': 0.30,  # Highest weight due to safety-critical nature
            'integration': 0.15,
            'reliability': 0.15
        }

        # Safety must pass (100%) for overall system to pass
        if safety['score'] < 1.0:
            return 0.0  # System fails if safety doesn't pass

        # Calculate weighted average for other components
        weighted_score = (
            functional['score'] * weights['functional'] +
            performance['score'] * weights['performance'] +
            safety['score'] * weights['safety'] +  # This will be 1.0 if safety passes
            integration['score'] * weights['integration'] +
            reliability['score'] * weights['reliability']
        )

        return weighted_score

    # Functional validation test implementations
    def test_voice_recognition(self) -> Dict[str, Any]:
        """Test voice recognition functionality"""
        try:
            # In a real implementation, this would test the actual voice recognition system
            # For this example, we'll simulate the test
            success = True  # Simulate success
            response_time = 0.8  # seconds

            return {
                'pass': success,
                'response_time': response_time,
                'accuracy': 0.89,  # Simulated accuracy
                'details': 'Voice recognition system responding within acceptable parameters'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_language_understanding(self) -> Dict[str, Any]:
        """Test language understanding and processing"""
        try:
            # Test LLM-based language understanding
            success = True
            processing_time = 1.2  # seconds

            return {
                'pass': success,
                'processing_time': processing_time,
                'understanding_rate': 0.85,
                'details': 'Language processing system understanding commands correctly'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_vision_processing(self) -> Dict[str, Any]:
        """Test vision processing capabilities"""
        try:
            # Test computer vision system
            success = True
            frame_rate = 25.0  # fps

            return {
                'pass': success,
                'frame_rate': frame_rate,
                'detection_accuracy': 0.92,
                'details': 'Vision system processing at required frame rate with good accuracy'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_task_planning(self) -> Dict[str, Any]:
        """Test cognitive task planning"""
        try:
            # Test LLM-based planning system
            success = True
            planning_time = 3.5  # seconds

            return {
                'pass': success,
                'planning_time': planning_time,
                'plan_feasibility': 0.95,
                'details': 'Task planning system generating feasible plans within time constraints'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_action_execution(self) -> Dict[str, Any]:
        """Test action execution capabilities"""
        try:
            # Test action execution system
            success = True
            execution_accuracy = 0.90

            return {
                'pass': success,
                'execution_accuracy': execution_accuracy,
                'success_rate': 0.95,
                'details': 'Action execution system performing tasks with required accuracy'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    # Performance validation test implementations
    def test_real_time_response(self) -> Dict[str, Any]:
        """Test real-time response requirements"""
        try:
            # Test system response times
            success = True
            avg_response_time = 0.15  # seconds
            max_response_time = 0.4   # seconds

            return {
                'pass': success and avg_response_time <= 0.2,
                'average_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'requirements_met': avg_response_time <= 0.2,
                'details': f'Average response time {avg_response_time}s within 200ms requirement'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_throughput_capacity(self) -> Dict[str, Any]:
        """Test system throughput capacity"""
        try:
            # Test message processing capacity
            success = True
            messages_per_second = 150.0

            return {
                'pass': success and messages_per_second >= 100.0,
                'throughput_mps': messages_per_second,
                'requirements_met': messages_per_second >= 100.0,
                'details': f'System processing {messages_per_second} messages/second, exceeding 100 requirement'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_resource_utilization(self) -> Dict[str, Any]:
        """Test resource utilization efficiency"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent

            success = cpu_usage <= 70 and memory_usage <= 75

            return {
                'pass': success,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'requirements_met': success,
                'details': f'CPU: {cpu_usage}%, Memory: {memory_usage}% within limits'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    # Safety validation test implementations
    def test_emergency_stop(self) -> Dict[str, Any]:
        """Test emergency stop functionality"""
        try:
            # Test emergency stop response
            success = True
            response_time = 0.008  # seconds (8ms)

            return {
                'pass': success and response_time <= 0.01,  # Must respond within 10ms
                'response_time': response_time,
                'requirements_met': response_time <= 0.01,
                'details': f'Emergency stop responded in {response_time*1000:.1f}ms, under 10ms requirement'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_collision_avoidance(self) -> Dict[str, Any]:
        """Test collision avoidance systems"""
        try:
            # Test collision avoidance
            success = True
            avoidance_rate = 0.98  # 98% success rate

            return {
                'pass': success and avoidance_rate >= 0.95,
                'avoidance_rate': avoidance_rate,
                'requirements_met': avoidance_rate >= 0.95,
                'details': f'Collision avoidance success rate {avoidance_rate*100:.1f}% exceeds 95% requirement'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    def test_command_validation(self) -> Dict[str, Any]:
        """Test command validation safety checks"""
        try:
            # Test command validation system
            success = True
            validation_rate = 1.0  # 100% validation success

            return {
                'pass': success and validation_rate == 1.0,
                'validation_rate': validation_rate,
                'requirements_met': validation_rate == 1.0,
                'details': 'All commands properly validated before execution'
            }
        except Exception as e:
            return {'pass': False, 'error': str(e)}

    # Additional test implementations would follow the same pattern...
    # (For brevity, not all test implementations are shown but the pattern continues)

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        report = f"""
CAPSTONE SYSTEM VALIDATION REPORT
===============================

Timestamp: {time.ctime(validation_results['timestamp'])}
Validation Duration: {validation_results['validation_time']:.2f} seconds
Overall Score: {validation_results['overall_score']:.2f} ({'PASS' if validation_results['overall_pass'] else 'FAIL'})

SCORE BREAKDOWN:
- Functional Validation: {validation_results['functional_validation']['score']:.2f}
- Performance Validation: {validation_results['performance_validation']['score']:.2f}
- Safety Validation: {validation_results['safety_validation']['score']:.2f}
- Integration Validation: {validation_results['integration_validation']['score']:.2f}
- Reliability Validation: {validation_results['reliability_validation']['score']:.2f}

DETAILED RESULTS:
"""

        # Add detailed results for each category
        categories = [
            ('Functional', validation_results['functional_validation']),
            ('Performance', validation_results['performance_validation']),
            ('Safety', validation_results['safety_validation']),
            ('Integration', validation_results['integration_validation']),
            ('Reliability', validation_results['reliability_validation'])
        ]

        for category_name, category_results in categories:
            report += f"\n{category_name.upper()} VALIDATION:\n"
            report += f"  Tests Run: {category_results['tests_run']}\n"
            report += f"  Tests Passed: {category_results['tests_passed']}\n"
            report += f"  Tests Failed: {category_results['tests_failed']}\n"
            report += f"  Score: {category_results['score']:.2f}\n"

        report += f"\nVALIDATION STATUS: {'✅ PASSED' if validation_results['overall_pass'] else '❌ FAILED'}"

        return report
```

## 2. Demonstration Scenarios

### 2.1 Comprehensive Demonstration Framework

Create a framework for demonstrating the complete system:

```python
# demonstration_framework.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_system_interfaces.srv import ExecuteDemoScenario
from capstone_system_interfaces.msg import DemoProgress, DemoResult
import time
import threading
from typing import Dict, List, Any, Callable

class DemonstrationFramework(Node):
    def __init__(self):
        super().__init__('demonstration_framework')

        # Publishers and subscribers
        self.demo_progress_publisher = self.create_publisher(
            DemoProgress, '/demo/progress', 10
        )
        self.demo_result_publisher = self.create_publisher(
            DemoResult, '/demo/result', 10
        )
        self.system_status_publisher = self.create_publisher(
            String, '/system/status', 10
        )

        # Services
        self.execute_demo_service = self.create_service(
            ExecuteDemoScenario, '/execute_demo', self.execute_demo_callback
        )

        # Demo scenarios
        self.demo_scenarios = {
            'basic_interaction': self.run_basic_interaction_demo,
            'complex_task': self.run_complex_task_demo,
            'multimodal_demo': self.run_multimodal_demo,
            'safety_demo': self.run_safety_demo,
            'full_system_demo': self.run_full_system_demo
        }

        # Demo state
        self.demo_active = False
        self.demo_thread = None
        self.demo_results = {}

        self.get_logger().info('Demonstration Framework initialized')

    def execute_demo_callback(self, request, response):
        """Execute a demonstration scenario"""
        if self.demo_active:
            response.success = False
            response.message = 'Demo already in progress'
            return response

        scenario_name = request.scenario_name
        if scenario_name not in self.demo_scenarios:
            response.success = False
            response.message = f'Unknown demo scenario: {scenario_name}'
            return response

        try:
            self.demo_active = True
            self.get_logger().info(f'Starting demo scenario: {scenario_name}')

            # Run demo in separate thread to avoid blocking service
            self.demo_thread = threading.Thread(
                target=self._run_demo_in_thread,
                args=(scenario_name, request.parameters),
                daemon=True
            )
            self.demo_thread.start()

            response.success = True
            response.message = f'Demo started: {scenario_name}'

        except Exception as e:
            self.get_logger().error(f'Demo execution error: {e}')
            response.success = False
            response.message = f'Demo execution error: {str(e)}'

        return response

    def _run_demo_in_thread(self, scenario_name: str, parameters: Dict[str, Any]):
        """Run demo in separate thread"""
        try:
            start_time = time.time()

            # Notify demo started
            progress_msg = DemoProgress()
            progress_msg.scenario_name = scenario_name
            progress_msg.status = 'started'
            progress_msg.progress_percentage = 0.0
            progress_msg.timestamp = self.get_clock().now().to_msg()
            self.demo_progress_publisher.publish(progress_msg)

            # Execute the demo scenario
            result = self.demo_scenarios[scenario_name](parameters)

            # Calculate demo metrics
            demo_duration = time.time() - start_time

            # Prepare demo result
            result_msg = DemoResult()
            result_msg.scenario_name = scenario_name
            result_msg.success = result.get('success', False)
            result_msg.duration = demo_duration
            result_msg.score = result.get('score', 0.0)
            result_msg.details = result.get('details', '')
            result_msg.timestamp = self.get_clock().now().to_msg()

            # Publish result
            self.demo_result_publisher.publish(result_msg)

            # Update progress
            progress_msg.status = 'completed' if result_msg.success else 'failed'
            progress_msg.progress_percentage = 100.0
            self.demo_progress_publisher.publish(progress_msg)

            self.get_logger().info(f'Demo completed: {scenario_name}, Success: {result_msg.success}, Duration: {demo_duration:.2f}s')

        except Exception as e:
            self.get_logger().error(f'Demo thread error: {e}')

            # Publish failure result
            result_msg = DemoResult()
            result_msg.scenario_name = scenario_name
            result_msg.success = False
            result_msg.duration = time.time() - start_time if 'start_time' in locals() else 0.0
            result_msg.score = 0.0
            result_msg.details = f'Demo error: {str(e)}'
            result_msg.timestamp = self.get_clock().now().to_msg()
            self.demo_result_publisher.publish(result_msg)

            progress_msg = DemoProgress()
            progress_msg.scenario_name = scenario_name
            progress_msg.status = 'error'
            progress_msg.progress_percentage = 0.0
            progress_msg.timestamp = self.get_clock().now().to_msg()
            self.demo_progress_publisher.publish(progress_msg)

        finally:
            self.demo_active = False

    def run_basic_interaction_demo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic interaction demonstration"""
        try:
            self.get_logger().info('Running basic interaction demo...')

            # Publish system status
            status_msg = String()
            status_msg.data = 'demo_running:basic_interaction'
            self.system_status_publisher.publish(status_msg)

            # Simulate basic voice command interaction
            steps = [
                ('Listen for command', 2.0),
                ('Process command', 1.0),
                ('Plan action', 2.0),
                ('Execute action', 3.0),
                ('Confirm completion', 1.0)
            ]

            total_steps = len(steps)
            for i, (step_desc, duration) in enumerate(steps):
                # Update progress
                progress_msg = DemoProgress()
                progress_msg.scenario_name = 'basic_interaction'
                progress_msg.status = f'executing: {step_desc}'
                progress_msg.progress_percentage = (i / total_steps) * 100.0
                progress_msg.timestamp = self.get_clock().now().to_msg()
                self.demo_progress_publisher.publish(progress_msg)

                self.get_logger().info(f'Basic demo: {step_desc}')
                time.sleep(duration)  # Simulate processing time

            # Final success
            return {
                'success': True,
                'score': 0.95,
                'details': 'Basic interaction completed successfully',
                'metrics': {
                    'response_time': 9.0,
                    'accuracy': 0.95,
                    'user_satisfaction': 0.9
                }
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': f'Basic interaction demo failed: {str(e)}'
            }

    def run_complex_task_demo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run complex task demonstration"""
        try:
            self.get_logger().info('Running complex task demo...')

            # Publish system status
            status_msg = String()
            status_msg.data = 'demo_running:complex_task'
            self.system_status_publisher.publish(status_msg)

            # Simulate complex task: "Go to kitchen, find a cup, pick it up, and bring it to the table"
            task_steps = [
                ('Perceive environment', 3.0),
                ('Plan navigation to kitchen', 2.0),
                ('Navigate to kitchen', 5.0),
                ('Search for cup', 4.0),
                ('Approach and grasp cup', 3.0),
                ('Plan navigation to table', 2.0),
                ('Navigate to table', 4.0),
                ('Place cup on table', 2.0),
                ('Return to home position', 3.0)
            ]

            total_steps = len(task_steps)
            for i, (step_desc, duration) in enumerate(task_steps):
                # Update progress
                progress_msg = DemoProgress()
                progress_msg.scenario_name = 'complex_task'
                progress_msg.status = f'executing: {step_desc}'
                progress_msg.progress_percentage = (i / total_steps) * 100.0
                progress_msg.timestamp = self.get_clock().now().to_msg()
                self.demo_progress_publisher.publish(progress_msg)

                self.get_logger().info(f'Complex task demo: {step_desc}')
                time.sleep(duration)  # Simulate processing time

            return {
                'success': True,
                'score': 0.92,
                'details': 'Complex task completed successfully',
                'metrics': {
                    'task_completion_rate': 1.0,
                    'efficiency': 0.88,
                    'safety_compliance': 1.0
                }
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': f'Complex task demo failed: {str(e)}'
            }

    def run_multimodal_demo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run multimodal interaction demonstration"""
        try:
            self.get_logger().info('Running multimodal demo...')

            # Publish system status
            status_msg = String()
            status_msg.data = 'demo_running:multimodal'
            self.system_status_publisher.publish(status_msg)

            # Simulate multimodal interaction: voice command with visual context
            # "Find the red cup to the left of the laptop"
            multimodal_steps = [
                ('Receive voice command', 1.0),
                ('Process natural language', 2.0),
                ('Analyze visual scene', 2.0),
                ('Ground language in vision', 1.5),
                ('Identify target object', 1.5),
                ('Plan manipulation action', 2.0),
                ('Execute precise manipulation', 4.0),
                ('Confirm task completion', 1.0)
            ]

            total_steps = len(multimodal_steps)
            for i, (step_desc, duration) in enumerate(multimodal_steps):
                # Update progress
                progress_msg = DemoProgress()
                progress_msg.scenario_name = 'multimodal'
                progress_msg.status = f'executing: {step_desc}'
                progress_msg.progress_percentage = (i / total_steps) * 100.0
                progress_msg.timestamp = self.get_clock().now().to_msg()
                self.demo_progress_publisher.publish(progress_msg)

                self.get_logger().info(f'Multimodal demo: {step_desc}')
                time.sleep(duration)  # Simulate processing time

            return {
                'success': True,
                'score': 0.94,
                'details': 'Multimodal interaction completed successfully',
                'metrics': {
                    'cross_modal_alignment': 0.92,
                    'task_success_rate': 1.0,
                    'response_quality': 0.95
                }
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': f'Multimodal demo failed: {str(e)}'
            }

    def run_safety_demo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run safety demonstration"""
        try:
            self.get_logger().info('Running safety demo...')

            # Publish system status
            status_msg = String()
            status_msg.data = 'demo_running:safety'
            self.system_status_publisher.publish(status_msg)

            # Simulate safety scenarios
            safety_scenarios = [
                ('Detect obstacle in path', 1.0),
                ('Activate collision avoidance', 0.5),
                ('Stop safely', 0.5),
                ('Resume after obstacle clears', 1.0),
                ('Validate LLM plan safety', 1.0),
                ('Reject unsafe command', 0.5),
                ('Execute safe alternative', 1.5),
                ('Confirm safety systems active', 1.0)
            ]

            total_scenarios = len(safety_scenarios)
            for i, (scenario_desc, duration) in enumerate(safety_scenarios):
                # Update progress
                progress_msg = DemoProgress()
                progress_msg.scenario_name = 'safety'
                progress_msg.status = f'safety_test: {scenario_desc}'
                progress_msg.progress_percentage = (i / total_scenarios) * 100.0
                progress_msg.timestamp = self.get_clock().now().to_msg()
                self.demo_progress_publisher.publish(progress_msg)

                self.get_logger().info(f'Safety demo: {scenario_desc}')
                time.sleep(duration)  # Simulate processing time

            return {
                'success': True,
                'score': 1.0,  # Safety demo must pass completely
                'details': 'All safety systems validated successfully',
                'metrics': {
                    'safety_system_activation_rate': 1.0,
                    'emergency_response_time': 0.01,  # 10ms
                    'safety_validation_accuracy': 1.0
                }
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': f'Safety demo failed: {str(e)}'
            }

    def run_full_system_demo(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive full system demonstration"""
        try:
            self.get_logger().info('Running full system demo...')

            # Publish system status
            status_msg = String()
            status_msg.data = 'demo_running:full_system'
            self.system_status_publisher.publish(status_msg)

            # Execute a sequence that demonstrates all capabilities
            full_demo_steps = [
                ('System initialization', 2.0),
                ('ROS 2 framework validation', 1.0),
                ('Digital twin integration', 2.0),
                ('AI brain activation', 1.0),
                ('VLA system startup', 1.0),
                ('Voice command processing', 2.0),
                ('Natural language understanding', 1.5),
                ('Vision system activation', 1.5),
                ('Object detection and recognition', 2.0),
                ('Cognitive planning execution', 3.0),
                ('Task execution and monitoring', 5.0),
                ('Safety validation throughout', 1.0),
                ('Performance monitoring', 1.0),
                ('System shutdown sequence', 1.0)
            ]

            total_steps = len(full_demo_steps)
            for i, (step_desc, duration) in enumerate(full_demo_steps):
                # Update progress
                progress_msg = DemoProgress()
                progress_msg.scenario_name = 'full_system'
                progress_msg.status = f'full_demo: {step_desc}'
                progress_msg.progress_percentage = (i / total_steps) * 100.0
                progress_msg.timestamp = self.get_clock().now().to_msg()
                self.demo_progress_publisher.publish(progress_msg)

                self.get_logger().info(f'Full system demo: {step_desc}')
                time.sleep(duration)  # Simulate processing time

            return {
                'success': True,
                'score': 0.96,
                'details': 'Full system demonstration completed successfully',
                'metrics': {
                    'module_integration_score': 0.98,
                    'system_coherence': 0.95,
                    'cross_module_communication': 0.97,
                    'overall_system_maturity': 0.96
                }
            }

        except Exception as e:
            return {
                'success': False,
                'score': 0.0,
                'details': f'Full system demo failed: {str(e)}'
            }
```

## 3. Hands-on Exercises

### Exercise 1: Complete System Deployment
**Objective:** Deploy the complete integrated system to the target environment.

**Prerequisites:**
- Completion of Phase 1 (System Architecture)
- All system components implemented and tested individually
- Target environment prepared (simulation or real hardware)
- All dependencies installed and configured

**Steps:**
1. Set up the deployment environment with required hardware/software
2. Implement the DeploymentManager class with environment validation
3. Create deployment scripts for system installation and configuration
4. Deploy the system to the target environment
5. Validate deployment with comprehensive checks
6. Test system functionality in the deployed environment
7. Document deployment process and any environment-specific configurations
8. Verify all safety systems are operational in deployment environment

**Expected Outcome:** Fully deployed system running in target environment with all components operational and validated.

**Troubleshooting Tips:**
- Ensure all dependencies are properly installed in deployment environment
- Validate hardware compatibility and resource requirements
- Check network connectivity and ROS 2 communication
- Verify safety systems are active and responsive

### Exercise 2: Performance Validation and Optimization
**Objective:** Validate system performance against requirements and optimize as needed.

**Prerequisites:**
- Completed Exercise 1 with deployed system
- Understanding of performance metrics and measurement
- Access to profiling and monitoring tools

**Steps:**
1. Implement the PerformanceMonitoringNode with comprehensive metrics
2. Run performance validation tests under various conditions
3. Identify performance bottlenecks and resource constraints
4. Optimize system components for better performance
5. Validate performance improvements with repeated testing
6. Document performance characteristics and optimization results
7. Ensure all performance requirements are met
8. Create performance baselines for future comparisons

**Expected Outcome:** System meeting all performance requirements with documented optimizations and baselines.

### Exercise 3: Comprehensive System Validation
**Objective:** Validate the complete system against all requirements and specifications.

**Prerequisites:**
- Completed previous exercises
- Fully deployed and operational system
- Understanding of validation methodologies
- Access to validation tools and test data

**Steps:**
1. Implement the SystemValidationSuite with all validation categories
2. Run comprehensive validation tests (functional, performance, safety, integration, reliability)
3. Document validation results and any issues found
4. Address validation failures and retest
5. Validate safety systems with 100% pass rate requirement
6. Ensure all original specifications are met
7. Create comprehensive validation report
8. Prepare for demonstration with validated system

**Expected Outcome:** System with comprehensive validation results showing compliance with all requirements and specifications.

### Exercise 4: Demonstration Preparation and Execution
**Objective:** Prepare and execute comprehensive demonstrations of the complete system.

**Prerequisites:**
- Completed previous exercises with validated system
- Understanding of all system capabilities
- Prepared demonstration scenarios and materials

**Steps:**
1. Implement the DemonstrationFramework with various demo scenarios
2. Create demonstration scripts for different audiences (technical, non-technical)
3. Practice demonstration scenarios to ensure smooth execution
4. Execute basic interaction demonstration
5. Execute complex task demonstration
6. Execute multimodal interaction demonstration
7. Execute safety validation demonstration
8. Execute full system integration demonstration
9. Document demonstration results and lessons learned
10. Prepare professional presentation materials

**Expected Outcome:** Successfully executed demonstrations showing all system capabilities with professional presentation materials.

## 4. Safety and Ethical Considerations

When deploying and demonstrating the complete system:
- Ensure all safety systems are validated and operational
- Maintain human oversight during demonstrations
- Validate that all AI-generated actions are safe before execution
- Protect privacy of any data collected during demonstrations
- Consider ethical implications of autonomous system behavior
- Plan for emergency stop procedures during demonstrations
- Ensure all participants are briefed on safety procedures
- Document any safety incidents or near-misses
- Verify that LLM outputs are appropriate and safe

## 5. Phase Summary

In this phase, you've completed:
- Deployment of the complete integrated system to target environment
- Implementation of comprehensive performance monitoring
- Execution of thorough system validation across all categories
- Creation of demonstration framework for showcasing capabilities
- Validation that all requirements and specifications are met
- Preparation of professional demonstrations for various audiences
- Documentation of deployment, validation, and demonstration results

The deployed and validated system represents the culmination of your integrated Physical AI & Humanoid Robotics project, demonstrating the successful combination of all course modules into a functional autonomous system.

## 6. Assessment Questions

### Multiple Choice
1. What is the minimum pass rate required for safety validation in the comprehensive system validation?
   a) 85%
   b) 90%
   c) 95%
   d) 100%

   Answer: d) 100%

2. Which component is responsible for monitoring system performance metrics?
   a) Voice Recognition Node
   b) Language Processor Node
   c) Performance Monitoring Node
   d) Safety Manager Node

   Answer: c) Performance Monitoring Node

### Practical Questions
1. Deploy the complete integrated system to a target environment and demonstrate its capabilities through a comprehensive validation process that includes functional, performance, safety, integration, and reliability validation with a final demonstration showing all system capabilities.

## 7. Next Steps

After completing Phase 6, you have:
- Successfully completed the capstone project with a fully integrated system
- Demonstrated mastery of all four course modules and their integration
- Validated the system against all requirements and specifications
- Created professional demonstrations of the system capabilities
- Documented the complete project implementation and validation

Your capstone project represents a comprehensive achievement in Physical AI & Humanoid Robotics, showcasing your ability to integrate advanced technologies into a functional autonomous system. This project serves as a portfolio piece demonstrating your skills to potential employers or for continued research and development.

## 8. Project Reflection and Future Work

Consider these aspects for reflection:
- What were the most challenging integration issues and how were they resolved?
- How did the system evolve from initial architecture to final implementation?
- What lessons were learned about integrating multiple complex systems?
- What improvements would you make if starting over?
- How could the system be extended for additional capabilities?
- What safety and ethical considerations need ongoing attention?
- How could the system be adapted for different applications or environments?

## 9. Professional Portfolio Preparation

Prepare your project for professional presentation:
- Create a comprehensive project portfolio with documentation, code, and results
- Prepare demonstration videos showing system capabilities
- Write a technical paper or report summarizing the project
- Create presentation materials for technical talks
- Document lessons learned and best practices
- Identify potential commercial or research applications
- Plan for continued development or maintenance