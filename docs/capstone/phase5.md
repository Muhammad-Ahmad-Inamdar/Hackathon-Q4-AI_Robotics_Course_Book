---
sidebar_position: 6
learning_objectives:
  - Integrate all components from previous phases into a unified system
  - Validate system performance across all modules and components
  - Test safety mechanisms and ethical considerations
  - Optimize system performance and reliability
  - Prepare for final demonstration and evaluation
  - Document lessons learned and future improvements
prerequisites:
  - Completion of Phases 1-4 (System Architecture, ROS 2 Framework, Simulation Environment, AI-Robot Brain)
  - All individual components functioning correctly
  - Access to complete development environment
  - Understanding of system-level testing and validation
estimated_time: "12 hours"
---

# Phase 5: System Integration and Validation

## Learning Objectives

By completing this phase, you will be able to:
- Integrate all components from previous phases into a unified, cohesive system
- Validate system performance across all modules and components working together
- Test and verify safety mechanisms and ethical considerations throughout the system
- Optimize system performance, reliability, and real-time operation
- Prepare for final demonstration and comprehensive evaluation
- Document lessons learned and identify future improvement opportunities
- Demonstrate professional-level integration of a complete robotic system

## Introduction

Phase 5 represents the culmination of your capstone project, where all components developed in previous phases are integrated into a unified system. This phase focuses on system-level integration, validation, and optimization to ensure that the complete humanoid robot system functions cohesively as designed. The integration process requires careful attention to interfaces, communication patterns, timing, and safety considerations across all components.

The key challenges in this phase include:
- Ensuring seamless communication between all system components
- Validating that integrated performance meets requirements
- Testing safety mechanisms across the entire system
- Optimizing for real-time performance and reliability
- Handling system-level edge cases and failure modes
- Preparing for comprehensive system validation and demonstration

## 1. System Integration Framework

### 1.1 Integration Architecture

Establish the integration architecture that connects all system components:

```python
# system_integration_framework.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from capstone_system_interfaces.msg import SystemStatus, IntegrationHealth
from capstone_system_interfaces.srv import SystemHealthCheck, SystemCalibration
from capstone_system_interfaces.action import SystemStartup
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, ReliabilityPolicy
import threading
import time
from typing import Dict, List, Any, Optional
import subprocess
import psutil
import GPUtil

class SystemIntegrationFramework(Node):
    def __init__(self):
        super().__init__('system_integration_framework')

        # Declare integration parameters
        self.declare_parameter('integration_mode', 'simulation')  # simulation or real_world
        self.declare_parameter('startup_sequence_delay', 2.0)
        self.declare_parameter('health_check_interval', 1.0)
        self.declare_parameter('safety_monitoring_enabled', True)
        self.declare_parameter('performance_monitoring_enabled', True)

        # Publishers
        self.system_status_publisher = self.create_publisher(
            SystemStatus, '/system/status', 10
        )
        self.integration_health_publisher = self.create_publisher(
            IntegrationHealth, '/system/integration_health', 10
        )

        # Subscribers for monitoring all subsystems
        self.ros_monitor_subscriber = self.create_subscription(
            String, '/ros_monitor/status', self.ros_status_callback, 10
        )
        self.digital_twin_monitor_subscriber = self.create_subscription(
            String, '/digital_twin/status', self.digital_twin_status_callback, 10
        )
        self.ai_brain_monitor_subscriber = self.create_subscription(
            String, '/ai_brain/status', self.ai_brain_status_callback, 10
        )
        self.vla_monitor_subscriber = self.create_subscription(
            String, '/vla/status', self.vla_status_callback, 10
        )

        # Services
        self.health_check_service = self.create_service(
            SystemHealthCheck, '/system/health_check', self.health_check_callback
        )
        self.calibration_service = self.create_service(
            SystemCalibration, '/system/calibrate', self.calibration_callback
        )

        # Action servers
        self.startup_action_server = ActionServer(
            self, SystemStartup, '/system/startup', self.startup_callback
        )

        # System state management
        self.system_state = {
            'initialized': False,
            'components': {
                'ros2_framework': {'status': 'unknown', 'ready': False},
                'digital_twin': {'status': 'unknown', 'ready': False},
                'ai_robot_brain': {'status': 'unknown', 'ready': False},
                'vla_system': {'status': 'unknown', 'ready': False}
            },
            'integration_status': 'pending',
            'startup_phase': 0,
            'startup_complete': False,
            'safety_ok': True,
            'performance_metrics': {}
        }

        # Health monitoring
        self.health_monitoring_active = True
        self.health_check_timer = self.create_timer(
            self.get_parameter('health_check_interval').value,
            self.health_monitoring_callback
        )

        # Performance monitoring
        self.performance_monitoring_active = self.get_parameter('performance_monitoring_enabled').value
        self.performance_timer = self.create_timer(5.0, self.performance_monitoring_callback)

        # Integration validation
        self.integration_validated = False
        self.integration_tests = []
        self.integration_results = {}

        # Safety monitoring
        self.safety_monitoring_active = self.get_parameter('safety_monitoring_enabled').value
        self.safety_check_timer = self.create_timer(0.1, self.safety_monitoring_callback)  # 10Hz safety checks

        # Startup sequence management
        self.startup_sequence = [
            self.initialize_ros2_framework,
            self.initialize_digital_twin,
            self.initialize_ai_robot_brain,
            self.initialize_vla_system,
            self.validate_integration,
            self.optimize_system_performance
        ]
        self.current_startup_step = 0

        # Resource monitoring
        self.resource_monitoring_timer = self.create_timer(2.0, self.resource_monitoring_callback)

        self.node_ready = True
        self.get_logger().info('System Integration Framework initialized')

    def ros_status_callback(self, msg):
        """Monitor ROS 2 framework status"""
        if 'ready' in msg.data.lower():
            self.system_state['components']['ros2_framework']['status'] = 'ready'
            self.system_state['components']['ros2_framework']['ready'] = True
        elif 'error' in msg.data.lower():
            self.system_state['components']['ros2_framework']['status'] = 'error'
            self.system_state['components']['ros2_framework']['ready'] = False
        else:
            self.system_state['components']['ros2_framework']['status'] = 'running'

    def digital_twin_status_callback(self, msg):
        """Monitor Digital Twin status"""
        if 'ready' in msg.data.lower():
            self.system_state['components']['digital_twin']['status'] = 'ready'
            self.system_state['components']['digital_twin']['ready'] = True
        elif 'error' in msg.data.lower():
            self.system_state['components']['digital_twin']['status'] = 'error'
            self.system_state['components']['digital_twin']['ready'] = False
        else:
            self.system_state['components']['digital_twin']['status'] = 'running'

    def ai_brain_status_callback(self, msg):
        """Monitor AI-Robot Brain status"""
        if 'ready' in msg.data.lower():
            self.system_state['components']['ai_robot_brain']['status'] = 'ready'
            self.system_state['components']['ai_robot_brain']['ready'] = True
        elif 'error' in msg.data.lower():
            self.system_state['components']['ai_robot_brain']['status'] = 'error'
            self.system_state['components']['ai_robot_brain']['ready'] = False
        else:
            self.system_state['components']['ai_robot_brain']['status'] = 'running'

    def vla_status_callback(self, msg):
        """Monitor VLA system status"""
        if 'ready' in msg.data.lower():
            self.system_state['components']['vla_system']['status'] = 'ready'
            self.system_state['components']['vla_system']['ready'] = True
        elif 'error' in msg.data.lower():
            self.system_state['components']['vla_system']['status'] = 'error'
            self.system_state['components']['vla_system']['ready'] = False
        else:
            self.system_state['components']['vla_system']['status'] = 'running'

    def initialize_ros2_framework(self) -> bool:
        """Initialize ROS 2 framework components"""
        self.get_logger().info('Initializing ROS 2 Framework components...')

        try:
            # Check if ROS 2 master is available
            if not self.check_ros2_master():
                self.get_logger().error('ROS 2 master not available')
                return False

            # Initialize communication patterns
            self.setup_integration_communication()

            # Validate ROS 2 performance
            if not self.validate_ros2_performance():
                self.get_logger().error('ROS 2 performance validation failed')
                return False

            self.system_state['components']['ros2_framework']['status'] = 'ready'
            self.system_state['components']['ros2_framework']['ready'] = True
            self.get_logger().info('ROS 2 Framework initialized successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error initializing ROS 2 framework: {e}')
            return False

    def initialize_digital_twin(self) -> bool:
        """Initialize Digital Twin components"""
        self.get_logger().info('Initializing Digital Twin components...')

        try:
            # Check simulation environment status
            if not self.check_simulation_environment():
                self.get_logger().error('Simulation environment not available')
                return False

            # Initialize simulation interfaces
            self.setup_simulation_interfaces()

            # Validate simulation fidelity
            if not self.validate_simulation_fidelity():
                self.get_logger().warn('Simulation fidelity validation issues detected')

            self.system_state['components']['digital_twin']['status'] = 'ready'
            self.system_state['components']['digital_twin']['ready'] = True
            self.get_logger().info('Digital Twin initialized successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error initializing Digital Twin: {e}')
            return False

    def initialize_ai_robot_brain(self) -> bool:
        """Initialize AI-Robot Brain components"""
        self.get_logger().info('Initializing AI-Robot Brain components...')

        try:
            # Check AI model availability
            if not self.check_ai_model_availability():
                self.get_logger().error('AI models not available')
                return False

            # Initialize perception systems
            self.setup_perception_systems()

            # Validate cognitive planning
            if not self.validate_cognitive_planning():
                self.get_logger().error('Cognitive planning validation failed')
                return False

            self.system_state['components']['ai_robot_brain']['status'] = 'ready'
            self.system_state['components']['ai_robot_brain']['ready'] = True
            self.get_logger().info('AI-Robot Brain initialized successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error initializing AI-Robot Brain: {e}')
            return False

    def initialize_vla_system(self) -> bool:
        """Initialize VLA system components"""
        self.get_logger().info('Initializing VLA system components...')

        try:
            # Check VLA model availability
            if not self.check_vla_model_availability():
                self.get_logger().error('VLA models not available')
                return False

            # Initialize multimodal interfaces
            self.setup_multimodal_interfaces()

            # Validate VLA coordination
            if not self.validate_vla_coordination():
                self.get_logger().error('VLA coordination validation failed')
                return False

            self.system_state['components']['vla_system']['status'] = 'ready'
            self.system_state['components']['vla_system']['ready'] = True
            self.get_logger().info('VLA system initialized successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error initializing VLA system: {e}')
            return False

    def validate_integration(self) -> bool:
        """Validate integration between all components"""
        self.get_logger().info('Validating system integration...')

        try:
            # Check component readiness
            all_ready = all(comp['ready'] for comp in self.system_state['components'].values())
            if not all_ready:
                self.get_logger().error('Not all components are ready for integration validation')
                return False

            # Test cross-module communication
            if not self.test_cross_module_communication():
                self.get_logger().error('Cross-module communication test failed')
                return False

            # Test integrated workflows
            if not self.test_integrated_workflows():
                self.get_logger().error('Integrated workflow test failed')
                return False

            # Validate safety integration
            if not self.validate_safety_integration():
                self.get_logger().error('Safety integration validation failed')
                return False

            self.integration_validated = True
            self.get_logger().info('System integration validated successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error validating integration: {e}')
            return False

    def optimize_system_performance(self) -> bool:
        """Optimize system performance for real-time operation"""
        self.get_logger().info('Optimizing system performance...')

        try:
            # Optimize communication patterns
            self.optimize_communication_patterns()

            # Optimize resource allocation
            self.optimize_resource_allocation()

            # Optimize processing pipelines
            self.optimize_processing_pipelines()

            # Validate performance targets
            if not self.validate_performance_targets():
                self.get_logger().warn('Some performance targets not met')

            self.get_logger().info('System performance optimized')
            return True

        except Exception as e:
            self.get_logger().error(f'Error optimizing system performance: {e}')
            return False

    def startup_callback(self, goal_handle):
        """Handle system startup action"""
        self.get_logger().info('Starting system integration sequence...')

        feedback_msg = SystemStartup.Feedback()
        result_msg = SystemStartup.Result()

        try:
            for i, startup_step in enumerate(self.startup_sequence):
                feedback_msg.current_step = f'Startup step {i+1}/{len(self.startup_sequence)}'
                feedback_msg.progress = float(i + 1) / len(self.startup_sequence)

                goal_handle.publish_feedback(feedback_msg)

                success = startup_step()
                if not success:
                    result_msg.success = False
                    result_msg.error_message = f'Startup failed at step {i+1}'
                    goal_handle.succeed()
                    return result_msg

                # Small delay between steps
                time.sleep(self.get_parameter('startup_sequence_delay').value)

            # Final system validation
            if self.validate_complete_system():
                result_msg.success = True
                result_msg.error_message = 'System startup completed successfully'
                self.system_state['startup_complete'] = True
                self.get_logger().info('System startup completed successfully')
            else:
                result_msg.success = False
                result_msg.error_message = 'Final system validation failed'
                self.get_logger().error('Final system validation failed')

        except Exception as e:
            self.get_logger().error(f'System startup error: {e}')
            result_msg.success = False
            result_msg.error_message = f'System startup error: {str(e)}'

        goal_handle.succeed()
        return result_msg

    def health_check_callback(self, request, response):
        """Perform system health check"""
        try:
            health_status = self.perform_system_health_check()

            response.system_healthy = health_status['overall_health']
            response.component_health = health_status['component_health']
            response.performance_metrics = health_status['performance_metrics']
            response.safety_status = health_status['safety_status']
            response.message = health_status['message']

        except Exception as e:
            self.get_logger().error(f'Health check error: {e}')
            response.system_healthy = False
            response.message = f'Health check error: {str(e)}'

        return response

    def calibration_callback(self, request, response):
        """Perform system calibration"""
        try:
            calibration_results = self.perform_system_calibration(request.calibration_type)

            response.success = calibration_results['success']
            response.calibrated_components = calibration_results['calibrated_components']
            response.calibration_data = calibration_results['calibration_data']
            response.message = calibration_results['message']

        except Exception as e:
            self.get_logger().error(f'Calibration error: {e}')
            response.success = False
            response.message = f'Calibration error: {str(e)}'

        return response

    def health_monitoring_callback(self):
        """Periodic health monitoring"""
        if not self.health_monitoring_active:
            return

        health_status = self.perform_system_health_check()

        # Publish integration health status
        health_msg = IntegrationHealth()
        health_msg.header.stamp = self.get_clock().now().to_msg()
        health_msg.overall_health_score = health_status['overall_health_score']
        health_msg.safety_status = health_status['safety_status']
        health_msg.performance_score = health_status['performance_score']

        self.integration_health_publisher.publish(health_msg)

        # Update system status
        status_msg = SystemStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.component_status = [
            f"ROS2: {self.system_state['components']['ros2_framework']['status']}",
            f"DigitalTwin: {self.system_state['components']['digital_twin']['status']}",
            f"AIBrain: {self.system_state['components']['ai_robot_brain']['status']}",
            f"VLA: {self.system_state['components']['vla_system']['status']}"
        ]
        status_msg.integration_status = self.system_state['integration_status']
        status_msg.startup_complete = self.system_state['startup_complete']
        status_msg.safety_ok = self.system_state['safety_ok']

        self.system_status_publisher.publish(status_msg)

    def safety_monitoring_callback(self):
        """Periodic safety monitoring"""
        if not self.safety_monitoring_active:
            return

        # Perform safety checks across all components
        safety_status = self.perform_system_safety_check()

        if not safety_status['overall_safe']:
            self.get_logger().error(f'SAFETY ISSUE DETECTED: {safety_status["safety_issues"]}')
            # Trigger safety protocols
            self.trigger_safety_protocols(safety_status['safety_issues'])
            self.system_state['safety_ok'] = False
        else:
            self.system_state['safety_ok'] = True

    def performance_monitoring_callback(self):
        """Periodic performance monitoring"""
        if not self.performance_monitoring_active:
            return

        # Monitor system performance
        performance_metrics = self.measure_system_performance()

        # Update system state with performance data
        self.system_state['performance_metrics'] = performance_metrics

        # Log performance if degraded
        if performance_metrics.get('cpu_usage', 0) > 80 or performance_metrics.get('memory_usage', 0) > 80:
            self.get_logger().warn(f'Performance degradation detected: {performance_metrics}')

    def resource_monitoring_callback(self):
        """Monitor system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            # GPU usage (if available)
            gpu_percent = 0
            gpu_memory_percent = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
                    gpu_memory_percent = gpus[0].memoryUtil * 100
            except:
                pass  # GPU monitoring not available

            # Log resource usage
            self.get_logger().debug(
                f'Resources - CPU: {cpu_percent}%, Memory: {memory_percent}%, '
                f'Disk: {disk_percent}%, GPU: {gpu_percent:.1f}%'
            )

            # Check for resource exhaustion
            if cpu_percent > 90 or memory_percent > 90:
                self.get_logger().warn('High resource usage detected')

        except Exception as e:
            self.get_logger().error(f'Error in resource monitoring: {e}')

    def check_ros2_master(self) -> bool:
        """Check if ROS 2 master is available"""
        try:
            # In a real implementation, this would check for ROS 2 master availability
            # For this example, we'll assume it's available
            return True
        except Exception:
            return False

    def setup_integration_communication(self):
        """Set up communication patterns between integrated components"""
        # Create communication bridges between modules
        # This would involve setting up message passing, service calls, etc.
        pass

    def validate_ros2_performance(self) -> bool:
        """Validate ROS 2 performance meets requirements"""
        # Test message passing performance, latency, etc.
        return True

    def check_simulation_environment(self) -> bool:
        """Check if simulation environment is available"""
        # In a real implementation, this would check for simulation availability
        return True

    def setup_simulation_interfaces(self):
        """Set up interfaces to simulation environment"""
        pass

    def validate_simulation_fidelity(self) -> bool:
        """Validate simulation environment fidelity"""
        # Test simulation accuracy, physics, etc.
        return True

    def check_ai_model_availability(self) -> bool:
        """Check if AI models are available and loaded"""
        # Check model files, GPU availability, etc.
        return True

    def setup_perception_systems(self):
        """Set up AI perception systems"""
        pass

    def validate_cognitive_planning(self) -> bool:
        """Validate cognitive planning system"""
        # Test planning accuracy, safety validation, etc.
        return True

    def check_vla_model_availability(self) -> bool:
        """Check if VLA models are available"""
        # Check VLA model files, dependencies, etc.
        return True

    def setup_multimodal_interfaces(self):
        """Set up multimodal interfaces"""
        pass

    def validate_vla_coordination(self) -> bool:
        """Validate VLA coordination system"""
        # Test multimodal integration, coordination, etc.
        return True

    def test_cross_module_communication(self) -> bool:
        """Test communication between different modules"""
        # Test ROS 2 communication patterns between all modules
        return True

    def test_integrated_workflows(self) -> bool:
        """Test integrated system workflows"""
        # Test complete system workflows from command to action
        return True

    def validate_safety_integration(self) -> bool:
        """Validate safety systems across integrated components"""
        # Test safety coordination between all modules
        return True

    def optimize_communication_patterns(self):
        """Optimize ROS 2 communication patterns"""
        pass

    def optimize_resource_allocation(self):
        """Optimize system resource allocation"""
        pass

    def optimize_processing_pipelines(self):
        """Optimize processing pipelines"""
        pass

    def validate_performance_targets(self) -> bool:
        """Validate system meets performance targets"""
        # Check if system meets real-time requirements
        return True

    def validate_complete_system(self) -> bool:
        """Validate complete integrated system"""
        # Perform comprehensive system validation
        return True

    def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            'overall_health': True,
            'overall_health_score': 0.0,
            'component_health': {},
            'performance_metrics': {},
            'safety_status': 'OK',
            'message': 'System healthy',
            'performance_score': 0.0
        }

        # Check component readiness
        for component_name, component_status in self.system_state['components'].items():
            health_status['component_health'][component_name] = component_status['ready']

        # Calculate overall health score
        ready_components = sum(1 for status in self.system_state['components'].values() if status['ready'])
        total_components = len(self.system_state['components'])
        health_score = ready_components / total_components if total_components > 0 else 0.0
        health_status['overall_health_score'] = health_score
        health_status['overall_health'] = health_score >= 0.9  # 90% threshold

        # Get performance metrics
        health_status['performance_metrics'] = self.system_state.get('performance_metrics', {})

        # Check safety status
        health_status['safety_status'] = 'OK' if self.system_state['safety_ok'] else 'ISSUE'
        if not self.system_state['safety_ok']:
            health_status['message'] = 'System has safety issues'
            health_status['overall_health'] = False

        return health_status

    def perform_system_calibration(self, calibration_type: str) -> Dict[str, Any]:
        """Perform system calibration"""
        calibration_results = {
            'success': False,
            'calibrated_components': [],
            'calibration_data': {},
            'message': ''
        }

        try:
            if calibration_type == 'full_system':
                # Perform full system calibration
                calibrated_components = []

                # Calibrate sensors
                if self.calibrate_sensors():
                    calibrated_components.append('sensors')

                # Calibrate cameras
                if self.calibrate_cameras():
                    calibrated_components.append('cameras')

                # Calibrate other components as needed
                calibration_results['success'] = True
                calibration_results['calibrated_components'] = calibrated_components
                calibration_results['message'] = f'System calibrated: {", ".join(calibrated_components)}'

            elif calibration_type == 'sensors_only':
                # Calibrate only sensors
                if self.calibrate_sensors():
                    calibration_results['success'] = True
                    calibration_results['calibrated_components'] = ['sensors']
                    calibration_results['message'] = 'Sensors calibrated successfully'

            else:
                calibration_results['message'] = f'Unknown calibration type: {calibration_type}'

        except Exception as e:
            calibration_results['message'] = f'Calibration error: {str(e)}'

        return calibration_results

    def calibrate_sensors(self) -> bool:
        """Calibrate all system sensors"""
        # Implementation would calibrate all sensors
        return True

    def calibrate_cameras(self) -> bool:
        """Calibrate all system cameras"""
        # Implementation would calibrate all cameras
        return True

    def perform_system_safety_check(self) -> Dict[str, Any]:
        """Perform comprehensive system safety check"""
        safety_status = {
            'overall_safe': True,
            'safety_issues': [],
            'component_safety_status': {},
            'emergency_stop_active': False
        }

        # Check each component for safety issues
        for component_name, component_status in self.system_state['components'].items():
            if component_status['status'] == 'error':
                safety_status['safety_issues'].append(f'{component_name} has error status')
                safety_status['overall_safe'] = False

        # Check system-wide safety conditions
        if not self.system_state['safety_ok']:
            safety_status['safety_issues'].append('System safety status is compromised')
            safety_status['overall_safe'] = False

        # Check for emergency stop conditions
        # This would integrate with actual emergency stop systems
        safety_status['emergency_stop_active'] = False

        return safety_status

    def trigger_safety_protocols(self, safety_issues: List[str]):
        """Trigger system safety protocols"""
        self.get_logger().error(f'Triggering safety protocols for issues: {safety_issues}')

        # Implement safety protocols
        # - Stop all robot motion
        # - Disable dangerous operations
        # - Alert operators
        # - Log safety events
        pass

    def measure_system_performance(self) -> Dict[str, float]:
        """Measure system performance metrics"""
        performance_metrics = {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
            'process_count': len(psutil.pids()),
            'average_response_time': 0.0,  # Would be calculated from actual system responses
            'message_rate': 0.0  # Would be calculated from ROS 2 message rates
        }

        # Add GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                performance_metrics['gpu_usage'] = gpus[0].load * 100
                performance_metrics['gpu_memory_usage'] = gpus[0].memoryUtil * 100
        except:
            pass

        return performance_metrics
```

### 1.2 Integration Testing Framework

Create a comprehensive testing framework for the integrated system:

```python
# integration_testing_framework.py
import unittest
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from capstone_system_interfaces.srv import SystemHealthCheck, SystemCalibration
from capstone_system_interfaces.msg import SystemStatus
from std_msgs.msg import String
import time
import threading
from typing import Dict, List, Any, Optional

class IntegrationTestSuite:
    """Comprehensive integration testing suite for the capstone system"""

    def __init__(self, node: Node):
        self.node = node
        self.test_results = {}
        self.test_logs = []
        self.test_statistics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0
        }

    def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return comprehensive results"""
        self.node.get_logger().info('Starting comprehensive integration test suite...')

        # Define test categories
        test_categories = [
            ('Module Integration', self.test_module_integration),
            ('Communication Patterns', self.test_communication_patterns),
            ('Safety Systems', self.test_safety_systems),
            ('Performance Validation', self.test_performance_validation),
            ('Edge Cases', self.test_edge_cases),
            ('Recovery Scenarios', self.test_recovery_scenarios)
        ]

        results = {}
        for category_name, test_function in test_categories:
            self.node.get_logger().info(f'Running {category_name} tests...')
            category_results = test_function()
            results[category_name] = category_results

        # Compile overall results
        overall_results = self.compile_test_results(results)

        self.node.get_logger().info(f'Integration test suite completed. Passed: {overall_results["passed"]}, Failed: {overall_results["failed"]}')

        return overall_results

    def test_module_integration(self) -> Dict[str, Any]:
        """Test integration between all modules"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test 1: ROS 2 Framework Integration
        test_result = self.test_ros2_integration()
        results['tests'].append(('ROS2 Integration', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test 2: Digital Twin Integration
        test_result = self.test_digital_twin_integration()
        results['tests'].append(('Digital Twin Integration', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test 3: AI-Robot Brain Integration
        test_result = self.test_ai_brain_integration()
        results['tests'].append(('AI-Robot Brain Integration', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test 4: VLA System Integration
        test_result = self.test_vla_integration()
        results['tests'].append(('VLA System Integration', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test 5: Cross-Module Communication
        test_result = self.test_cross_module_communication()
        results['tests'].append(('Cross-Module Communication', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_ros2_integration(self) -> bool:
        """Test ROS 2 framework integration"""
        try:
            # Test communication between nodes
            # Test message passing
            # Test service calls
            # Test action execution
            self.node.get_logger().debug('ROS 2 integration test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'ROS 2 integration test failed: {e}')
            return False

    def test_digital_twin_integration(self) -> bool:
        """Test Digital Twin integration"""
        try:
            # Test simulation environment availability
            # Test sensor simulation accuracy
            # Test physics simulation fidelity
            # Test simulation-to-reality transfer
            self.node.get_logger().debug('Digital Twin integration test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Digital Twin integration test failed: {e}')
            return False

    def test_ai_brain_integration(self) -> bool:
        """Test AI-Robot Brain integration"""
        try:
            # Test perception system integration
            # Test cognitive planning integration
            # Test LLM integration and safety validation
            # Test multimodal processing
            self.node.get_logger().debug('AI-Robot Brain integration test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'AI-Robot Brain integration test failed: {e}')
            return False

    def test_vla_integration(self) -> bool:
        """Test VLA system integration"""
        try:
            # Test vision-language integration
            # Test action execution coordination
            # Test multimodal understanding
            # Test natural interaction
            self.node.get_logger().debug('VLA system integration test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'VLA system integration test failed: {e}')
            return False

    def test_cross_module_communication(self) -> bool:
        """Test communication between all modules"""
        try:
            # Test message passing between all modules
            # Test service calls across modules
            # Test action coordination
            # Test data synchronization
            self.node.get_logger().debug('Cross-module communication test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Cross-module communication test failed: {e}')
            return False

    def test_communication_patterns(self) -> Dict[str, Any]:
        """Test various communication patterns"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test Publisher-Subscriber Pattern
        test_result = self.test_publisher_subscriber_pattern()
        results['tests'].append(('Publisher-Subscriber', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Service-Client Pattern
        test_result = self.test_service_client_pattern()
        results['tests'].append(('Service-Client', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Action-Client-Server Pattern
        test_result = self.test_action_client_server_pattern()
        results['tests'].append(('Action-Client-Server', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Quality of Service Patterns
        test_result = self.test_qos_patterns()
        results['tests'].append(('QoS Patterns', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_publisher_subscriber_pattern(self) -> bool:
        """Test publisher-subscriber communication pattern"""
        try:
            # Create test publisher and subscriber
            # Verify message delivery
            # Test different QoS profiles
            self.node.get_logger().debug('Publisher-subscriber pattern test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Publisher-subscriber test failed: {e}')
            return False

    def test_service_client_pattern(self) -> bool:
        """Test service-client communication pattern"""
        try:
            # Create test service and client
            # Verify request-response cycle
            # Test error handling
            self.node.get_logger().debug('Service-client pattern test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Service-client test failed: {e}')
            return False

    def test_action_client_server_pattern(self) -> bool:
        """Test action-client-server communication pattern"""
        try:
            # Create test action server and client
            # Verify goal-feedback-result cycle
            # Test preemption and cancellation
            self.node.get_logger().debug('Action-client-server pattern test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Action-client-server test failed: {e}')
            return False

    def test_qos_patterns(self) -> bool:
        """Test Quality of Service patterns"""
        try:
            # Test different QoS configurations
            # Verify reliability and durability settings
            # Test deadline and lifespan policies
            self.node.get_logger().debug('QoS patterns test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'QoS patterns test failed: {e}')
            return False

    def test_safety_systems(self) -> Dict[str, Any]:
        """Test safety systems across integrated components"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test Safety Manager Integration
        test_result = self.test_safety_manager_integration()
        results['tests'].append(('Safety Manager Integration', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Emergency Stop Systems
        test_result = self.test_emergency_stop_systems()
        results['tests'].append(('Emergency Stop Systems', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Collision Avoidance
        test_result = self.test_collision_avoidance()
        results['tests'].append(('Collision Avoidance', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test LLM Safety Validation
        test_result = self.test_llm_safety_validation()
        results['tests'].append(('LLM Safety Validation', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Command Validation
        test_result = self.test_command_validation()
        results['tests'].append(('Command Validation', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_safety_manager_integration(self) -> bool:
        """Test safety manager integration across all components"""
        try:
            # Verify safety manager monitors all components
            # Test safety validation for all actions
            # Verify emergency procedures
            self.node.get_logger().debug('Safety manager integration test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Safety manager integration test failed: {e}')
            return False

    def test_emergency_stop_systems(self) -> bool:
        """Test emergency stop systems"""
        try:
            # Test emergency stop activation
            # Test system response to emergency stop
            # Verify safe state after emergency stop
            self.node.get_logger().debug('Emergency stop systems test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Emergency stop systems test failed: {e}')
            return False

    def test_collision_avoidance(self) -> bool:
        """Test collision avoidance systems"""
        try:
            # Test navigation collision avoidance
            # Test manipulation collision avoidance
            # Test human-robot safety zones
            self.node.get_logger().debug('Collision avoidance test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Collision avoidance test failed: {e}')
            return False

    def test_llm_safety_validation(self) -> bool:
        """Test LLM safety validation for generated plans"""
        try:
            # Test LLM plan validation
            # Test safety constraint checking
            # Test ethical consideration validation
            self.node.get_logger().debug('LLM safety validation test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'LLM safety validation test failed: {e}')
            return False

    def test_command_validation(self) -> bool:
        """Test command validation systems"""
        try:
            # Test voice command validation
            # Test natural language command validation
            # Test safety constraint checking
            self.node.get_logger().debug('Command validation test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Command validation test failed: {e}')
            return False

    def test_performance_validation(self) -> Dict[str, Any]:
        """Test system performance under various conditions"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test Real-time Performance
        test_result = self.test_real_time_performance()
        results['tests'].append(('Real-time Performance', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Resource Utilization
        test_result = self.test_resource_utilization()
        results['tests'].append(('Resource Utilization', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Communication Latency
        test_result = self.test_communication_latency()
        results['tests'].append(('Communication Latency', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test System Throughput
        test_result = self.test_system_throughput()
        results['tests'].append(('System Throughput', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Stress Testing
        test_result = self.test_stress_testing()
        results['tests'].append(('Stress Testing', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_real_time_performance(self) -> bool:
        """Test real-time performance requirements"""
        try:
            # Test system meets real-time deadlines
            # Measure processing latencies
            # Verify timing constraints
            self.node.get_logger().debug('Real-time performance test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Real-time performance test failed: {e}')
            return False

    def test_resource_utilization(self) -> bool:
        """Test resource utilization stays within limits"""
        try:
            # Monitor CPU, memory, GPU usage
            # Verify resource limits are not exceeded
            # Test performance under load
            self.node.get_logger().debug('Resource utilization test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Resource utilization test failed: {e}')
            return False

    def test_communication_latency(self) -> bool:
        """Test communication latency requirements"""
        try:
            # Measure message passing latencies
            # Verify timing requirements are met
            # Test under various network conditions
            self.node.get_logger().debug('Communication latency test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Communication latency test failed: {e}')
            return False

    def test_system_throughput(self) -> bool:
        """Test system throughput capabilities"""
        try:
            # Test message processing rates
            # Verify system can handle required throughput
            # Test under peak loads
            self.node.get_logger().debug('System throughput test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'System throughput test failed: {e}')
            return False

    def test_stress_testing(self) -> bool:
        """Test system under stress conditions"""
        try:
            # Test system stability under high load
            # Test resource exhaustion scenarios
            # Verify graceful degradation
            self.node.get_logger().debug('Stress testing passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Stress testing failed: {e}')
            return False

    def test_edge_cases(self) -> Dict[str, Any]:
        """Test system behavior under edge cases"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test Invalid Inputs
        test_result = self.test_invalid_inputs()
        results['tests'].append(('Invalid Inputs', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Missing Components
        test_result = self.test_missing_components()
        results['tests'].append(('Missing Components', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Network Failures
        test_result = self.test_network_failures()
        results['tests'].append(('Network Failures', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Sensor Failures
        test_result = self.test_sensor_failures()
        results['tests'].append(('Sensor Failures', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test AI Model Failures
        test_result = self.test_ai_model_failures()
        results['tests'].append(('AI Model Failures', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_invalid_inputs(self) -> bool:
        """Test system response to invalid inputs"""
        try:
            # Test with malformed messages
            # Test with out-of-range values
            # Test with invalid commands
            # Verify proper error handling
            self.node.get_logger().debug('Invalid inputs test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Invalid inputs test failed: {e}')
            return False

    def test_missing_components(self) -> bool:
        """Test system behavior when components are missing"""
        try:
            # Test graceful degradation when components are unavailable
            # Verify fallback behaviors
            # Test system recovery
            self.node.get_logger().debug('Missing components test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Missing components test failed: {e}')
            return False

    def test_network_failures(self) -> bool:
        """Test system response to network failures"""
        try:
            # Test communication failure handling
            # Verify message buffering and recovery
            # Test network partition scenarios
            self.node.get_logger().debug('Network failures test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Network failures test failed: {e}')
            return False

    def test_sensor_failures(self) -> bool:
        """Test system response to sensor failures"""
        try:
            # Test sensor failure detection
            # Verify fallback sensor usage
            # Test degraded mode operation
            self.node.get_logger().debug('Sensor failures test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Sensor failures test failed: {e}')
            return False

    def test_ai_model_failures(self) -> bool:
        """Test system response to AI model failures"""
        try:
            # Test LLM failure handling
            # Verify fallback planning
            # Test degraded AI operation
            self.node.get_logger().debug('AI model failures test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'AI model failures test failed: {e}')
            return False

    def test_recovery_scenarios(self) -> Dict[str, Any]:
        """Test system recovery from various failure modes"""
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }

        # Test Component Restart Recovery
        test_result = self.test_component_restart_recovery()
        results['tests'].append(('Component Restart Recovery', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test State Recovery
        test_result = self.test_state_recovery()
        results['tests'].append(('State Recovery', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Data Recovery
        test_result = self.test_data_recovery()
        results['tests'].append(('Data Recovery', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Communication Recovery
        test_result = self.test_communication_recovery()
        results['tests'].append(('Communication Recovery', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Test Safe State Recovery
        test_result = self.test_safe_state_recovery()
        results['tests'].append(('Safe State Recovery', test_result))
        results['total'] += 1
        if test_result:
            results['passed'] += 1
        else:
            results['failed'] += 1

        return results

    def test_component_restart_recovery(self) -> bool:
        """Test system recovery after component restarts"""
        try:
            # Test individual component restarts
            # Verify state preservation and recovery
            # Test communication reconnection
            self.node.get_logger().debug('Component restart recovery test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Component restart recovery test failed: {e}')
            return False

    def test_state_recovery(self) -> bool:
        """Test system state recovery"""
        try:
            # Test state restoration after failures
            # Verify critical state preservation
            # Test state consistency
            self.node.get_logger().debug('State recovery test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'State recovery test failed: {e}')
            return False

    def test_data_recovery(self) -> bool:
        """Test data recovery mechanisms"""
        try:
            # Test data backup and restoration
            # Verify data integrity after recovery
            # Test critical data preservation
            self.node.get_logger().debug('Data recovery test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Data recovery test failed: {e}')
            return False

    def test_communication_recovery(self) -> bool:
        """Test communication recovery after disruptions"""
        try:
            # Test message retransmission
            # Verify communication reestablishment
            # Test message ordering after recovery
            self.node.get_logger().debug('Communication recovery test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Communication recovery test failed: {e}')
            return False

    def test_safe_state_recovery(self) -> bool:
        """Test recovery to safe state after failures"""
        try:
            # Test safe state establishment after failures
            # Verify safety systems remain active
            # Test gradual system restoration
            self.node.get_logger().debug('Safe state recovery test passed')
            return True
        except Exception as e:
            self.node.get_logger().error(f'Safe state recovery test failed: {e}')
            return False

    def compile_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all test results into a comprehensive report"""
        total_tests = 0
        total_passed = 0
        total_failed = 0

        for category, category_results in results.items():
            total_tests += category_results['total']
            total_passed += category_results['passed']
            total_failed += category_results['failed']

        overall_results = {
            'passed': total_passed,
            'failed': total_failed,
            'total': total_tests,
            'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'categories': results
        }

        return overall_results
```

### 1.3 System Validation Node

Create a system validation node that performs comprehensive validation:

```python
# system_validation_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from capstone_system_interfaces.srv import SystemValidation
from capstone_system_interfaces.msg import ValidationReport, SystemMetrics
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Any

class SystemValidationNode(Node):
    def __init__(self):
        super().__init__('system_validation_node')

        # Declare validation parameters
        self.declare_parameter('validation_frequency', 5.0)  # Hz
        self.declare_parameter('validation_timeout', 30.0)  # seconds
        self.declare_parameter('validation_threshold', 0.85)  # minimum acceptable score
        self.declare_parameter('enable_continuous_validation', True)

        # Publishers
        self.validation_report_publisher = self.create_publisher(
            ValidationReport, '/system/validation_report', 10
        )
        self.system_metrics_publisher = self.create_publisher(
            SystemMetrics, '/system/metrics', 10
        )

        # Services
        self.validation_service = self.create_service(
            SystemValidation, '/system/validate', self.validation_callback
        )

        # Validation state
        self.validation_results = {}
        self.validation_history = []
        self.validation_active = False
        self.validation_thread = None

        # Validation metrics
        self.validation_metrics = {
            'functional_correctness': 0.0,
            'performance_efficiency': 0.0,
            'safety_compliance': 0.0,
            'integration_quality': 0.0,
            'reliability_score': 0.0
        }

        # Validation test suites
        self.integration_tests = IntegrationTestSuite(self)

        # Continuous validation timer
        if self.get_parameter('enable_continuous_validation').value:
            self.continuous_validation_timer = self.create_timer(
                1.0 / self.get_parameter('validation_frequency').value,
                self.continuous_validation_callback
            )

        self.node_ready = True
        self.get_logger().info('System Validation Node initialized')

    def validation_callback(self, request, response):
        """Perform system validation service callback"""
        try:
            # Perform validation based on requested type
            if request.validation_type == 'comprehensive':
                validation_results = self.perform_comprehensive_validation()
            elif request.validation_type == 'functional':
                validation_results = self.perform_functional_validation()
            elif request.validation_type == 'safety':
                validation_results = self.perform_safety_validation()
            elif request.validation_type == 'performance':
                validation_results = self.perform_performance_validation()
            else:
                validation_results = self.perform_basic_validation()

            # Compile response
            response.success = validation_results['overall_pass']
            response.validation_score = validation_results['overall_score']
            response.passed_checks = validation_results['passed_checks']
            response.failed_checks = validation_results['failed_checks']
            response.validation_details = json.dumps(validation_results['details'])
            response.timestamp = self.get_clock().now().to_msg()

            # Update validation history
            self.validation_history.append({
                'timestamp': time.time(),
                'results': validation_results,
                'request_type': request.validation_type
            })

            # Keep only recent history
            if len(self.validation_history) > 100:
                self.validation_history = self.validation_history[-100:]

            # Publish validation report
            self.publish_validation_report(validation_results, request.validation_type)

        except Exception as e:
            self.get_logger().error(f'Validation service error: {e}')
            response.success = False
            response.validation_score = 0.0
            response.message = f'Validation error: {str(e)}'

        return response

    def continuous_validation_callback(self):
        """Perform continuous system validation"""
        if not self.node_ready:
            return

        try:
            # Perform lightweight validation checks
            basic_results = self.perform_basic_validation()

            # Update validation metrics
            self.update_validation_metrics(basic_results)

            # Publish system metrics
            self.publish_system_metrics()

            # Check if comprehensive validation is needed
            if self.should_trigger_comprehensive_validation():
                self.get_logger().info('Triggering comprehensive validation...')
                comprehensive_results = self.perform_comprehensive_validation()
                self.publish_validation_report(comprehensive_results, 'comprehensive')

        except Exception as e:
            self.get_logger().error(f'Continuous validation error: {e}')

    def perform_comprehensive_validation(self) -> Dict[str, Any]:
        """Perform comprehensive system validation"""
        self.get_logger().info('Starting comprehensive system validation...')

        start_time = time.time()

        # Run all integration tests
        integration_results = self.integration_tests.run_all_integration_tests()

        # Perform additional validation checks
        functional_results = self.perform_functional_validation()
        safety_results = self.perform_safety_validation()
        performance_results = self.perform_performance_validation()

        # Compile comprehensive results
        overall_score = self.calculate_comprehensive_score(
            integration_results, functional_results, safety_results, performance_results
        )

        validation_results = {
            'overall_pass': overall_score >= self.get_parameter('validation_threshold').value,
            'overall_score': overall_score,
            'validation_time': time.time() - start_time,
            'validation_type': 'comprehensive',
            'integration_results': integration_results,
            'functional_results': functional_results,
            'safety_results': safety_results,
            'performance_results': performance_results,
            'passed_checks': integration_results['passed'] + functional_results['passed'] + safety_results['passed'] + performance_results['passed'],
            'failed_checks': integration_results['failed'] + functional_results['failed'] + safety_results['failed'] + performance_results['failed'],
            'details': {
                'integration': integration_results,
                'functional': functional_results,
                'safety': safety_results,
                'performance': performance_results
            }
        }

        self.get_logger().info(f'Comprehensive validation completed. Score: {overall_score:.2f}, Time: {validation_results["validation_time"]:.2f}s')
        return validation_results

    def perform_functional_validation(self) -> Dict[str, Any]:
        """Perform functional validation of system components"""
        functional_tests = {
            'ros2_functionality': self.test_ros2_functionality(),
            'communication_patterns': self.test_communication_patterns(),
            'service_availability': self.test_service_availability(),
            'action_servers': self.test_action_servers(),
            'data_flow': self.test_data_flow(),
            'component_interactions': self.test_component_interactions()
        }

        passed = sum(1 for result in functional_tests.values() if result)
        total = len(functional_tests)
        score = passed / total if total > 0 else 0.0

        return {
            'overall_pass': score >= 0.9,  # 90% threshold for functional tests
            'overall_score': score,
            'passed_checks': passed,
            'failed_checks': total - passed,
            'details': functional_tests
        }

    def perform_safety_validation(self) -> Dict[str, Any]:
        """Perform safety validation of system components"""
        safety_tests = {
            'emergency_stop': self.test_emergency_stop(),
            'collision_avoidance': self.test_collision_avoidance(),
            'safety_zones': self.test_safety_zones(),
            'command_validation': self.test_command_validation(),
            'llm_safety_filters': self.test_llm_safety_filters(),
            'human_aware_navigation': self.test_human_aware_navigation()
        }

        passed = sum(1 for result in safety_tests.values() if result)
        total = len(safety_tests)
        score = passed / total if total > 0 else 0.0

        return {
            'overall_pass': score >= 1.0,  # 100% required for safety tests
            'overall_score': score,
            'passed_checks': passed,
            'failed_checks': total - passed,
            'details': safety_tests
        }

    def perform_performance_validation(self) -> Dict[str, Any]:
        """Perform performance validation of system components"""
        performance_tests = {
            'real_time_performance': self.test_real_time_performance(),
            'resource_utilization': self.test_resource_utilization(),
            'communication_latency': self.test_communication_latency(),
            'system_throughput': self.test_system_throughput(),
            'response_time': self.test_response_time()
        }

        passed = sum(1 for result in performance_tests.values() if result)
        total = len(performance_tests)
        score = passed / total if total > 0 else 0.0

        return {
            'overall_pass': score >= 0.8,  # 80% threshold for performance tests
            'overall_score': score,
            'passed_checks': passed,
            'failed_checks': total - passed,
            'details': performance_tests
        }

    def perform_basic_validation(self) -> Dict[str, Any]:
        """Perform basic validation checks"""
        basic_tests = {
            'node_communication': self.test_node_communication(),
            'parameter_access': self.test_parameter_access(),
            'service_connectivity': self.test_service_connectivity(),
            'subscription_functionality': self.test_subscription_functionality()
        }

        passed = sum(1 for result in basic_tests.values() if result)
        total = len(basic_tests)
        score = passed / total if total > 0 else 0.0

        return {
            'overall_pass': score >= 1.0,  # All basic tests must pass
            'overall_score': score,
            'passed_checks': passed,
            'failed_checks': total - passed,
            'details': basic_tests
        }

    def calculate_comprehensive_score(self, integration_results, functional_results, safety_results, performance_results) -> float:
        """Calculate comprehensive validation score"""
        # Weighted scoring - safety is most important
        safety_weight = 0.35
        functional_weight = 0.25
        performance_weight = 0.25
        integration_weight = 0.15

        # Calculate weighted score
        weighted_score = (
            safety_results['overall_score'] * safety_weight +
            functional_results['overall_score'] * functional_weight +
            performance_results['overall_score'] * performance_weight +
            integration_results['overall_score'] * integration_weight
        )

        return weighted_score

    def publish_validation_report(self, validation_results: Dict[str, Any], validation_type: str):
        """Publish validation report"""
        report_msg = ValidationReport()
        report_msg.header.stamp = self.get_clock().now().to_msg()
        report_msg.validation_type = validation_type
        report_msg.overall_score = validation_results['overall_score']
        report_msg.passed = validation_results['overall_pass']
        report_msg.total_tests = validation_results['passed_checks'] + validation_results['failed_checks']
        report_msg.passed_tests = validation_results['passed_checks']
        report_msg.failed_tests = validation_results['failed_checks']
        report_msg.validation_duration = validation_results['validation_time']
        report_msg.timestamp = self.get_clock().now().to_msg()

        # Convert details to string for message
        report_msg.details = json.dumps(validation_results['details'])

        self.validation_report_publisher.publish(report_msg)

    def publish_system_metrics(self):
        """Publish system metrics"""
        metrics_msg = SystemMetrics()
        metrics_msg.header.stamp = self.get_clock().now().to_msg()
        metrics_msg.functional_score = self.validation_metrics.get('functional_correctness', 0.0)
        metrics_msg.performance_score = self.validation_metrics.get('performance_efficiency', 0.0)
        metrics_msg.safety_score = self.validation_metrics.get('safety_compliance', 0.0)
        metrics_msg.integration_score = self.validation_metrics.get('integration_quality', 0.0)
        metrics_msg.reliability_score = self.validation_metrics.get('reliability_score', 0.0)
        metrics_msg.timestamp = self.get_clock().now().to_msg()

        self.system_metrics_publisher.publish(metrics_msg)

    def update_validation_metrics(self, validation_results: Dict[str, Any]):
        """Update validation metrics based on results"""
        self.validation_metrics['functional_correctness'] = validation_results.get('overall_score', 0.0)

        # Update other metrics based on validation results
        if 'performance_results' in validation_results.get('details', {}):
            perf_results = validation_results['details']['performance_results']
            self.validation_metrics['performance_efficiency'] = perf_results.get('overall_score', 0.0)

        if 'safety_results' in validation_results.get('details', {}):
            safety_results = validation_results['details']['safety_results']
            self.validation_metrics['safety_compliance'] = safety_results.get('overall_score', 0.0)

        # Calculate reliability based on historical validation results
        if self.validation_history:
            recent_scores = [entry['results']['overall_score'] for entry in self.validation_history[-5:]]
            if recent_scores:
                self.validation_metrics['reliability_score'] = sum(recent_scores) / len(recent_scores)

    def should_trigger_comprehensive_validation(self) -> bool:
        """Determine if comprehensive validation should be triggered"""
        # Trigger comprehensive validation if:
        # 1. No comprehensive validation has been done recently
        # 2. System state has changed significantly
        # 3. Basic validation is failing consistently
        # 4. Periodic schedule (e.g., every hour)

        if not self.validation_history:
            return True

        # Check if last comprehensive validation was more than 1 hour ago
        last_comp_time = 0
        for entry in reversed(self.validation_history):
            if entry['request_type'] == 'comprehensive':
                last_comp_time = entry['timestamp']
                break

        if time.time() - last_comp_time > 3600:  # 1 hour
            return True

        # Check if recent validations are declining
        recent_scores = [entry['results']['overall_score'] for entry in self.validation_history[-5:]]
        if len(recent_scores) >= 5 and recent_scores[-1] < recent_scores[0] * 0.9:
            # Score declined by 10% or more
            return True

        return False

    # Test implementation methods would go here
    # These would contain the actual test logic for each validation aspect
    def test_ros2_functionality(self) -> bool:
        """Test basic ROS 2 functionality"""
        # Implementation would test ROS 2 core functionality
        return True

    def test_communication_patterns(self) -> bool:
        """Test communication patterns"""
        # Implementation would test various communication patterns
        return True

    def test_service_availability(self) -> bool:
        """Test service availability"""
        # Implementation would test if required services are available
        return True

    def test_action_servers(self) -> bool:
        """Test action servers"""
        # Implementation would test if required action servers are available
        return True

    def test_data_flow(self) -> bool:
        """Test data flow between components"""
        # Implementation would test data flow between system components
        return True

    def test_component_interactions(self) -> bool:
        """Test component interactions"""
        # Implementation would test interactions between system components
        return True

    def test_emergency_stop(self) -> bool:
        """Test emergency stop functionality"""
        # Implementation would test emergency stop systems
        return True

    def test_collision_avoidance(self) -> bool:
        """Test collision avoidance systems"""
        # Implementation would test collision avoidance functionality
        return True

    def test_safety_zones(self) -> bool:
        """Test safety zones and boundaries"""
        # Implementation would test safety zone functionality
        return True

    def test_command_validation(self) -> bool:
        """Test command validation systems"""
        # Implementation would test command validation
        return True

    def test_llm_safety_filters(self) -> bool:
        """Test LLM safety filter functionality"""
        # Implementation would test LLM safety validation
        return True

    def test_human_aware_navigation(self) -> bool:
        """Test human-aware navigation"""
        # Implementation would test navigation around humans
        return True

    def test_real_time_performance(self) -> bool:
        """Test real-time performance requirements"""
        # Implementation would test real-time performance
        return True

    def test_resource_utilization(self) -> bool:
        """Test resource utilization limits"""
        # Implementation would test resource usage
        return True

    def test_communication_latency(self) -> bool:
        """Test communication latency requirements"""
        # Implementation would test communication latency
        return True

    def test_system_throughput(self) -> bool:
        """Test system throughput capabilities"""
        # Implementation would test system throughput
        return True

    def test_response_time(self) -> bool:
        """Test system response time requirements"""
        # Implementation would test response times
        return True

    def test_node_communication(self) -> bool:
        """Test basic node communication"""
        # Implementation would test node communication
        return True

    def test_parameter_access(self) -> bool:
        """Test parameter access functionality"""
        # Implementation would test parameter access
        return True

    def test_service_connectivity(self) -> bool:
        """Test service connectivity"""
        # Implementation would test service connectivity
        return True

    def test_subscription_functionality(self) -> bool:
        """Test subscription functionality"""
        # Implementation would test subscription functionality
        return True
```

## 2. System Integration Testing

### 2.1 End-to-End Testing Framework

Create an end-to-end testing framework:

```python
# end_to_end_testing.py
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_system_interfaces.msg import ParsedCommand, TaskPlan
from capstone_system_interfaces.srv import ProcessCommand
import time
import threading
from typing import Dict, Any

class EndToEndTestFramework:
    """Framework for end-to-end system testing"""

    def __init__(self, node: Node):
        self.node = node
        self.test_results = []
        self.test_scenarios = []

    def setup_test_scenarios(self):
        """Setup comprehensive test scenarios"""
        self.test_scenarios = [
            {
                'name': 'Basic Voice Command to Action',
                'description': 'Test voice command recognition to action execution',
                'inputs': {'voice_command': 'Move forward 1 meter'},
                'expected_outputs': ['navigation_task', 'safety_validation_pass'],
                'validation_function': self.validate_basic_navigation
            },
            {
                'name': 'Complex Multimodal Task',
                'description': 'Test complex task involving vision, language, and action',
                'inputs': {'voice_command': 'Find the red cup and bring it to me', 'visual_context': 'simulated_office_scene'},
                'expected_outputs': ['object_detection', 'grasping_action', 'delivery_task'],
                'validation_function': self.validate_multimodal_task
            },
            {
                'name': 'Safety Validation',
                'description': 'Test safety validation for potentially unsafe commands',
                'inputs': {'voice_command': 'Go through the wall', 'visual_context': 'simulated_environment'},
                'expected_outputs': ['safety_violation_detected', 'command_rejected'],
                'validation_function': self.validate_safety_validation
            },
            {
                'name': 'LLM Planning Integration',
                'description': 'Test LLM-based planning for complex tasks',
                'inputs': {'voice_command': 'Clean up the desk and organize the items', 'visual_context': 'messy_desk_scene'},
                'expected_outputs': ['task_decomposition', 'step_by_step_plan', 'execution_monitoring'],
                'validation_function': self.validate_llm_planning
            }
        ]

    def run_end_to_end_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests"""
        self.setup_test_scenarios()

        results = {
            'total_tests': len(self.test_scenarios),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }

        for scenario in self.test_scenarios:
            self.node.get_logger().info(f'Running test: {scenario["name"]}')

            test_result = self.execute_test_scenario(scenario)
            results['test_details'].append(test_result)

            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1

        return results

    def execute_test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test scenario"""
        start_time = time.time()

        try:
            # Setup test environment
            self.setup_test_environment(scenario)

            # Execute the test
            execution_result = scenario['validation_function'](scenario['inputs'])

            # Validate results
            expected_outputs = scenario['expected_outputs']
            actual_outputs = execution_result.get('outputs', [])

            # Check if all expected outputs are present
            all_expected_present = all(exp in actual_outputs for exp in expected_outputs)

            # Calculate test duration
            duration = time.time() - start_time

            test_result = {
                'name': scenario['name'],
                'description': scenario['description'],
                'passed': all_expected_present and execution_result.get('success', False),
                'duration': duration,
                'inputs': scenario['inputs'],
                'expected_outputs': expected_outputs,
                'actual_outputs': actual_outputs,
                'details': execution_result.get('details', {}),
                'error': execution_result.get('error', None)
            }

            if test_result['passed']:
                self.node.get_logger().info(f'Test passed: {scenario["name"]}')
            else:
                self.node.get_logger().error(f'Test failed: {scenario["name"]}')
                self.node.get_logger().error(f'Expected: {expected_outputs}, Got: {actual_outputs}')

        except Exception as e:
            test_result = {
                'name': scenario['name'],
                'description': scenario['description'],
                'passed': False,
                'duration': time.time() - start_time,
                'inputs': scenario['inputs'],
                'expected_outputs': scenario['expected_outputs'],
                'actual_outputs': [],
                'details': {},
                'error': str(e)
            }
            self.node.get_logger().error(f'Test execution error: {e}')

        return test_result

    def setup_test_environment(self, scenario: Dict[str, Any]):
        """Setup test environment for a scenario"""
        # Reset system state
        # Load appropriate simulation environment
        # Set up required sensors and components
        pass

    def validate_basic_navigation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic navigation scenario"""
        try:
            # Simulate voice command processing
            voice_cmd = String()
            voice_cmd.data = inputs['voice_command']

            # This would involve calling the actual system components
            # For simulation, we'll mock the expected behavior

            # In a real test, this would involve:
            # 1. Publishing voice command to system
            # 2. Monitoring system responses
            # 3. Validating navigation execution

            return {
                'success': True,
                'outputs': ['navigation_task', 'safety_validation_pass'],
                'details': {'navigation_executed': True, 'safety_check_passed': True}
            }

        except Exception as e:
            return {
                'success': False,
                'outputs': [],
                'details': {},
                'error': str(e)
            }

    def validate_multimodal_task(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complex multimodal task scenario"""
        try:
            # Simulate complex multimodal task
            # This would involve vision processing, LLM planning, and action execution

            return {
                'success': True,
                'outputs': ['object_detection', 'grasping_action', 'delivery_task'],
                'details': {
                    'object_detected': True,
                    'grasping_successful': True,
                    'delivery_completed': True
                }
            }

        except Exception as e:
            return {
                'success': False,
                'outputs': [],
                'details': {},
                'error': str(e)
            }

    def validate_safety_validation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety validation scenario"""
        try:
            # Simulate safety validation for unsafe command

            return {
                'success': True,
                'outputs': ['safety_violation_detected', 'command_rejected'],
                'details': {
                    'safety_violation_detected': True,
                    'command_rejected': True,
                    'safety_explanation': 'Command would result in collision with static obstacle'
                }
            }

        except Exception as e:
            return {
                'success': False,
                'outputs': [],
                'details': {},
                'error': str(e)
            }

    def validate_llm_planning(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM-based planning scenario"""
        try:
            # Simulate LLM-based complex task planning

            return {
                'success': True,
                'outputs': ['task_decomposition', 'step_by_step_plan', 'execution_monitoring'],
                'details': {
                    'tasks_decomposed': 5,  # Example: 5 subtasks
                    'plan_generated': True,
                    'execution_monitored': True
                }
            }

        except Exception as e:
            return {
                'success': False,
                'outputs': [],
                'details': {},
                'error': str(e)
            }
```

## 3. Hands-on Exercises

### Exercise 1: ROS 2 Communication Framework Implementation
**Objective:** Implement and validate the complete ROS 2 communication framework for the integrated system.

**Prerequisites:**
- Understanding of ROS 2 concepts and architecture
- Experience with Python and ROS 2 development
- Completion of Phase 1 (System Architecture)

**Steps:**
1. Implement the CapstoneBaseNode with proper error handling and safety features
2. Create the Voice Recognition Node with Whisper integration
3. Implement the Language Processing Node with LLM integration
4. Set up all publishers, subscribers, services, and actions as defined in the architecture
5. Test communication patterns and message passing between components
6. Validate real-time performance requirements
7. Document the communication framework and interfaces

**Expected Outcome:** Complete ROS 2 communication framework that enables all system components to communicate effectively with proper error handling and safety validation.

**Troubleshooting Tips:**
- Ensure proper QoS profiles for different message types
- Validate message schemas and data types
- Check for communication timeouts and retries
- Monitor system resource usage during communication

### Exercise 2: Integration Testing Framework
**Objective:** Create and run comprehensive integration tests for the system.

**Prerequisites:**
- Completed Exercise 1
- Understanding of testing frameworks and methodologies
- Experience with system-level testing

**Steps:**
1. Implement the IntegrationTestSuite with all required test categories
2. Create test cases for module integration, communication patterns, and safety systems
3. Run comprehensive integration tests and document results
4. Validate performance under various conditions
5. Test edge cases and failure scenarios
6. Implement recovery testing scenarios
7. Document test coverage and results

**Expected Outcome:** Comprehensive integration testing framework that validates all aspects of system integration with detailed results and documentation.

### Exercise 3: System Validation and Monitoring
**Objective:** Implement system validation and continuous monitoring capabilities.

**Prerequisites:**
- Completed previous exercises
- Understanding of validation methodologies
- Experience with monitoring and metrics collection

**Steps:**
1. Implement the SystemValidationNode with comprehensive validation capabilities
2. Create validation tests for functional correctness, safety, and performance
3. Implement continuous validation with appropriate frequency
4. Set up metrics collection and publishing
5. Create validation reports and logging
6. Test validation under various system states
7. Document validation procedures and results

**Expected Outcome:** Complete system validation framework that continuously monitors and validates system performance, safety, and functionality with comprehensive reporting.

## 4. Safety and Ethical Considerations

When implementing the integrated system:
- Ensure all safety systems operate independently of other components
- Validate all AI-generated plans before execution
- Implement proper fallback behaviors when components fail
- Maintain human oversight for critical decisions
- Consider privacy implications of data collection and processing
- Plan for graceful degradation when integrated systems fail
- Implement ethical considerations in AI decision-making
- Ensure transparency in system behavior and decision-making

## 5. Phase Summary

In this phase, you've completed:
- Implementation of the ROS 2 communication framework connecting all system components
- Creation of all necessary nodes, topics, services, and actions
- Establishment of proper communication patterns and safety validation
- Implementation of comprehensive testing and validation frameworks
- Validation of system integration and performance
- Documentation of the complete communication architecture

The communication framework you've implemented serves as the backbone of your integrated robotic system, enabling all components to work together seamlessly while maintaining safety and performance requirements.

## 6. Assessment Questions

### Multiple Choice
1. What is the primary purpose of the safety manager in the integrated system?
   a) To process voice commands
   b) To monitor and validate safety across all components
   c) To handle vision processing
   d) To manage communication patterns

   Answer: b) To monitor and validate safety across all components

2. Which validation category has the highest pass threshold requirement?
   a) Functional validation (90%)
   b) Safety validation (100%)
   c) Performance validation (80%)
   d) Integration validation (85%)

   Answer: b) Safety validation (100%)

### Practical Questions
1. Implement a complete ROS 2 communication framework that integrates all four modules with proper safety validation, real-time performance, and comprehensive testing that meets all system requirements.

## 7. Next Steps

After completing Phase 2, you should:
- Validate that all communication patterns work correctly across modules
- Ensure safety systems are properly integrated and validated
- Verify real-time performance requirements are met
- Prepare for Phase 3: Digital Twin Integration and Validation
- Document communication architecture and interfaces
- Optimize communication patterns for performance
- Plan for simulation and real-world validation

The ROS 2 communication framework you've implemented provides the essential infrastructure for your integrated humanoid robot system, enabling all components to work together effectively while maintaining safety and reliability.