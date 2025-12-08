---
sidebar_position: 3
learning_objectives:
  - Implement the ROS 2 communication framework for the capstone system
  - Create and configure ROS 2 nodes for all system components
  - Establish communication patterns and message passing between components
  - Implement service and action interfaces for complex behaviors
  - Validate and test the communication framework
prerequisites:
  - Completion of Phase 1 (System Architecture Design)
  - Strong understanding of ROS 2 concepts and programming
  - Experience with Python and ROS 2 development
  - System architecture specifications from Phase 1
estimated_time: "10 hours"
---

# Phase 2: ROS 2 Communication Framework Implementation

## Learning Objectives

By completing this phase, you will be able to:
- Implement the ROS 2 communication framework for the capstone system
- Create and configure ROS 2 nodes for all system components
- Establish communication patterns and message passing between components
- Implement service and action interfaces for complex behaviors
- Validate and test the communication framework for reliability and performance
- Ensure the communication framework meets safety and performance requirements

## Introduction

Phase 2 focuses on implementing the ROS 2 communication framework that will serve as the backbone of your capstone system. This phase brings to life the architecture designed in Phase 1 by creating the actual ROS 2 nodes, topics, services, and actions that will enable communication between all system components.

The ROS 2 framework implementation is critical because it:
- Provides the communication infrastructure for all modules
- Ensures proper message passing between system components
- Enables real-time coordination of the integrated system
- Supports the safety and performance requirements defined in the architecture
- Facilitates testing and validation of the integrated system

## 1. ROS 2 System Implementation

### 1.1 Core Node Implementation

Implement the core nodes for the system based on the architecture:

```python
# capstone_nodes.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, SetBool
from rclpy.action import ActionServer, ActionClient
from action_msgs.msg import GoalStatus
import json
import threading
import time
from typing import Dict, List, Any, Optional

class CapstoneBaseNode(Node):
    """Base class for all capstone system nodes"""

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # Initialize node parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('log_level', 'info')
        self.declare_parameter('config_file', '')

        # QoS profiles for different communication patterns
        self.qos_sensor_data = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.qos_control = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.qos_configuration = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Node state management
        self.node_ready = False
        self.system_status = {
            'initialized': False,
            'running': False,
            'safety_ok': True,
            'error_count': 0
        }

        self.get_logger().info(f'{node_name} initialized successfully')

    def safe_publish(self, publisher, message):
        """Safely publish a message with error handling"""
        try:
            if self.node_ready:
                publisher.publish(message)
        except Exception as e:
            self.get_logger().error(f'Failed to publish message: {e}')
            self.system_status['error_count'] += 1

    def safe_call_service(self, client, request, timeout_sec=5.0):
        """Safely call a service with timeout"""
        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            return future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None
```

### 1.2 Voice Recognition Node

Implement the voice recognition node that interfaces with Whisper:

```python
# voice_recognition_node.py
import speech_recognition as sr
import threading
from std_msgs.msg import String
from std_msgs.srv import Trigger
from rcl_interfaces.msg import ParameterDescriptor
import queue
import numpy as np
import io
from scipy.io import wavfile

class VoiceRecognitionNode(CapstoneBaseNode):
    def __init__(self):
        super().__init__('voice_recognition_node')

        # Initialize speech recognition components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Configuration parameters
        self.declare_parameter('language', 'en-US', ParameterDescriptor(description='Speech recognition language'))
        self.declare_parameter('energy_threshold', 300, ParameterDescriptor(description='Energy threshold for silence detection'))
        self.declare_parameter('dynamic_energy_threshold', True, ParameterDescriptor(description='Enable dynamic energy threshold'))
        self.declare_parameter('phrase_time_limit', 5.0, ParameterDescriptor(description='Phrase time limit in seconds'))

        # Publishers and subscribers
        self.voice_command_publisher = self.create_publisher(
            String, '/voice_commands', self.qos_control
        )

        # Services
        self.start_recognition_service = self.create_service(
            Trigger, '/start_voice_recognition', self.start_recognition_callback
        )
        self.stop_recognition_service = self.create_service(
            Trigger, '/stop_voice_recognition', self.stop_recognition_callback
        )

        # Internal state
        self.listening_active = False
        self.recognition_thread = None
        self.audio_queue = queue.Queue()

        # Configure recognizer
        self.recognizer.energy_threshold = self.get_parameter('energy_threshold').value
        self.recognizer.dynamic_energy_threshold = self.get_parameter('dynamic_energy_threshold').value

        self.node_ready = True
        self.get_logger().info('Voice Recognition Node initialized')

    def start_recognition_callback(self, request, response):
        """Start voice recognition service callback"""
        if not self.listening_active:
            self.start_listening()
            response.success = True
            response.message = 'Voice recognition started'
            self.get_logger().info('Voice recognition started')
        else:
            response.success = False
            response.message = 'Voice recognition already active'

        return response

    def stop_recognition_callback(self, request, response):
        """Stop voice recognition service callback"""
        if self.listening_active:
            self.stop_listening()
            response.success = True
            response.message = 'Voice recognition stopped'
            self.get_logger().info('Voice recognition stopped')
        else:
            response.success = False
            response.message = 'Voice recognition not active'

        return response

    def start_listening(self):
        """Start the voice recognition process"""
        self.listening_active = True
        self.recognition_thread = threading.Thread(target=self._recognition_worker)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()

    def stop_listening(self):
        """Stop the voice recognition process"""
        self.listening_active = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)

    def _recognition_worker(self):
        """Worker thread for continuous speech recognition"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.listening_active:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source,
                        timeout=1.0,
                        phrase_time_limit=self.get_parameter('phrase_time_limit').value
                    )

                # Recognize speech using Google's free API
                language = self.get_parameter('language').value
                text = self.recognizer.recognize_google(audio, language=language)

                if text:
                    # Publish recognized text
                    cmd_msg = String()
                    cmd_msg.data = text
                    self.safe_publish(self.voice_command_publisher, cmd_msg)
                    self.get_logger().info(f'Recognized: {text}')

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                self.get_logger().debug('Could not understand audio')
                continue
            except sr.RequestError as e:
                self.get_logger().error(f'Could not request results from speech recognition service: {e}')
                continue
            except Exception as e:
                self.get_logger().error(f'Unexpected error in recognition: {e}')
                continue

    def process_audio_data(self, audio_data: bytes) -> Optional[str]:
        """Process raw audio data through speech recognition"""
        try:
            # Convert bytes to audio data that speech_recognition can handle
            audio = sr.AudioData(audio_data, 16000, 2)  # Assuming 16kHz, 16-bit
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception as e:
            self.get_logger().error(f'Error processing audio data: {e}')
            return None
```

### 1.3 Language Processing Node

Implement the language processing node that uses LLMs for understanding:

```python
# language_processor_node.py
import openai
import json
import re
from std_msgs.msg import String
from std_msgs.srv import Trigger
from capstone_system_interfaces.srv import ProcessCommand
from capstone_system_interfaces.msg import ParsedCommand, TaskPlan

class LanguageProcessorNode(CapstoneBaseNode):
    def __init__(self):
        super().__init__('language_processor_node')

        # Configuration parameters
        self.declare_parameter('openai_api_key', '', ParameterDescriptor(description='OpenAI API key'))
        self.declare_parameter('model', 'gpt-3.5-turbo', ParameterDescriptor(description='LLM model to use'))
        self.declare_parameter('temperature', 0.3, ParameterDescriptor(description='LLM temperature'))
        self.declare_parameter('max_tokens', 500, ParameterDescriptor(description='Maximum tokens for LLM'))

        # Publishers
        self.parsed_command_publisher = self.create_publisher(
            ParsedCommand, '/parsed_commands', self.qos_control
        )
        self.task_plan_publisher = self.create_publisher(
            TaskPlan, '/task_plans', self.qos_control
        )
        self.system_response_publisher = self.create_publisher(
            String, '/system_responses', self.qos_control
        )

        # Subscribers
        self.voice_command_subscriber = self.create_subscription(
            String, '/voice_commands', self.voice_command_callback, self.qos_control
        )

        # Services
        self.process_command_service = self.create_service(
            ProcessCommand, '/process_command', self.process_command_callback
        )

        # Initialize OpenAI client
        api_key = self.get_parameter('openai_api_key').value
        if api_key:
            openai.api_key = api_key
            self.llm_model = self.get_parameter('model').value
            self.temperature = self.get_parameter('temperature').value
            self.max_tokens = self.get_parameter('max_tokens').value

            self.llm_initialized = True
            self.get_logger().info('LLM client initialized')
        else:
            self.get_logger().warn('No OpenAI API key provided, LLM functionality disabled')
            self.llm_initialized = False

        # Context management
        self.conversation_history = []
        self.context_window_size = 10  # Keep last 10 exchanges

        self.node_ready = True
        self.get_logger().info('Language Processor Node initialized')

    def voice_command_callback(self, msg):
        """Process incoming voice commands"""
        if not self.llm_initialized:
            self.get_logger().warn('LLM not initialized, skipping command processing')
            return

        try:
            # Process the voice command using LLM
            parsed_command = self.parse_command(msg.data)

            if parsed_command:
                # Publish parsed command
                self.safe_publish(self.parsed_command_publisher, parsed_command)

                # Generate system response
                response = self.generate_response(msg.data)
                if response:
                    response_msg = String()
                    response_msg.data = response
                    self.safe_publish(self.system_response_publisher, response_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')

    def process_command_callback(self, request, response):
        """Process command service callback"""
        if not self.llm_initialized:
            response.success = False
            response.message = 'LLM not initialized'
            return response

        try:
            parsed_command = self.parse_command(request.command)
            if parsed_command:
                response.success = True
                response.message = 'Command processed successfully'
                response.parsed_command = parsed_command

                # Also publish the parsed command
                self.safe_publish(self.parsed_command_publisher, parsed_command)
            else:
                response.success = False
                response.message = 'Failed to parse command'

        except Exception as e:
            self.get_logger().error(f'Error in process command service: {e}')
            response.success = False
            response.message = f'Error processing command: {str(e)}'

        return response

    def parse_command(self, command: str) -> Optional[ParsedCommand]:
        """Parse a natural language command using LLM"""
        if not self.llm_initialized:
            return None

        try:
            # Create a structured prompt for command parsing
            prompt = f"""
            Parse the following natural language command into structured format:

            Command: "{command}"

            Respond with a JSON object containing:
            {{
                "intent": "the main intent of the command (e.g., 'navigate', 'manipulate', 'search')",
                "action": "specific action to perform",
                "objects": ["list", "of", "relevant", "objects"],
                "locations": ["list", "of", "relevant", "locations"],
                "parameters": {{"key": "value"}},
                "confidence": float_between_0_and_1
            }}

            Be concise and accurate. Only respond with the JSON object.
            """

            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a command parser for a robotic system. Parse natural language commands into structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = completion.choices[0].message.content.strip()

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)

                # Create ParsedCommand message
                parsed_command = ParsedCommand()
                parsed_command.intent = parsed_data.get('intent', '')
                parsed_command.action = parsed_data.get('action', '')
                parsed_command.objects = parsed_data.get('objects', [])
                parsed_command.locations = parsed_data.get('locations', [])
                parsed_command.parameters = json.dumps(parsed_data.get('parameters', {}))
                parsed_command.confidence = parsed_data.get('confidence', 0.0)
                parsed_command.original_command = command
                parsed_command.timestamp = self.get_clock().now().to_msg()

                # Add to conversation history
                self.conversation_history.append({
                    'command': command,
                    'parsed': parsed_data,
                    'timestamp': time.time()
                })

                # Maintain context window size
                if len(self.conversation_history) > self.context_window_size:
                    self.conversation_history.pop(0)

                return parsed_command

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON decode error in command parsing: {e}')
        except Exception as e:
            self.get_logger().error(f'Error parsing command: {e}')

        return None

    def generate_response(self, command: str) -> Optional[str]:
        """Generate a natural language response to the command"""
        if not self.llm_initialized:
            return "I understand your command."

        try:
            # Create context from conversation history
            context = "\n".join([
                f"User: {exchange['command']}\nSystem: Understood and processing"
                for exchange in self.conversation_history[-3:]  # Last 3 exchanges
            ])

            prompt = f"""
            Given the following context and command, generate a polite and informative response:

            Context:
            {context}

            Command: "{command}"

            Generate a brief, friendly response acknowledging the command and indicating next steps.
            """

            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful robotic assistant. Respond politely to user commands."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=100
            )

            response_text = completion.choices[0].message.content.strip()
            return response_text

        except Exception as e:
            self.get_logger().error(f'Error generating response: {e}')
            return f"I've received your command: {command}. Processing now."

    def generate_task_plan(self, parsed_command: ParsedCommand) -> Optional[TaskPlan]:
        """Generate a detailed task plan from parsed command"""
        if not self.llm_initialized:
            return None

        try:
            prompt = f"""
            Generate a detailed task plan for the following parsed command:

            Intent: {parsed_command.intent}
            Action: {parsed_command.action}
            Objects: {parsed_command.objects}
            Locations: {parsed_command.locations}
            Parameters: {parsed_command.parameters}

            Respond with a JSON object containing a task plan:
            {{
                "id": "unique_task_plan_id",
                "description": "brief description of the plan",
                "tasks": [
                    {{
                        "id": "task_id",
                        "description": "what to do",
                        "action_type": "navigation|manipulation|perception|communication",
                        "parameters": {{"param1": "value1", ...}},
                        "dependencies": ["task_id_1", ...],
                        "priority": integer
                    }}
                ],
                "estimated_duration_seconds": float,
                "safety_considerations": ["consideration1", ...]
            }}
            """

            completion = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a task planner for a robotic system. Generate detailed, executable task plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.max_tokens
            )

            response_text = completion.choices[0].message.content.strip()

            # Extract JSON and create TaskPlan message
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                plan_data = json.loads(json_str)

                task_plan = TaskPlan()
                task_plan.plan_id = plan_data.get('id', '')
                task_plan.description = plan_data.get('description', '')
                task_plan.estimated_duration = plan_data.get('estimated_duration_seconds', 0.0)
                task_plan.safety_considerations = plan_data.get('safety_considerations', [])

                # Convert tasks to appropriate format
                for task_data in plan_data.get('tasks', []):
                    # This would depend on your specific TaskPlan message definition
                    pass

                return task_plan

        except Exception as e:
            self.get_logger().error(f'Error generating task plan: {e}')

        return None
```

### 1.4 Vision System Node

Implement the vision system node that processes visual data:

```python
# vision_system_node.py
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from capstone_system_interfaces.msg import ObjectDetectionArray, SceneDescription
from capstone_system_interfaces.srv import AnalyzeScene
from geometry_msgs.msg import Point
import threading
import queue

class VisionSystemNode(CapstoneBaseNode):
    def __init__(self):
        super().__init__('vision_system_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Configuration parameters
        self.declare_parameter('enable_object_detection', True)
        self.declare_parameter('enable_tracking', False)
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('max_detection_range', 5.0)

        # Publishers
        self.object_detections_publisher = self.create_publisher(
            ObjectDetectionArray, '/object_detections', self.qos_sensor_data
        )
        self.scene_description_publisher = self.create_publisher(
            SceneDescription, '/scene_descriptions', self.qos_control
        )

        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, self.qos_sensor_data
        )

        # Services
        self.analyze_scene_service = self.create_service(
            AnalyzeScene, '/analyze_scene', self.analyze_scene_callback
        )

        # Internal processing
        self.image_processing_queue = queue.Queue(maxsize=2)  # Limit queue size
        self.processing_thread = threading.Thread(target=self._image_processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Vision processing components
        self.enable_object_detection = self.get_parameter('enable_object_detection').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.max_detection_range = self.get_parameter('max_detection_range').value

        # Mock object detector (in real implementation, this would be YOLO, etc.)
        self.object_detector = MockObjectDetector()

        self.node_ready = True
        self.get_logger().info('Vision System Node initialized')

    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Add image to processing queue (non-blocking)
            try:
                self.image_processing_queue.put_nowait(msg)
            except queue.Full:
                # Drop oldest image if queue is full
                try:
                    self.image_processing_queue.get_nowait()
                    self.image_processing_queue.put_nowait(msg)
                except queue.Empty:
                    pass  # Queue is empty, just add the new one
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def _image_processing_worker(self):
        """Worker thread for image processing"""
        while rclpy.ok():
            try:
                # Get image from queue with timeout
                image_msg = self.image_processing_queue.get(timeout=1.0)

                # Process image
                self._process_image(image_msg)

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Error in image processing worker: {e}')

    def _process_image(self, image_msg):
        """Process a single image message"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            results = {}

            # Perform object detection if enabled
            if self.enable_object_detection:
                detections = self.object_detector.detect(cv_image, self.detection_threshold)
                results['detections'] = detections

                # Publish object detections
                detection_msg = self._create_detection_message(detections, image_msg.header)
                self.safe_publish(self.object_detections_publisher, detection_msg)

            # Perform scene analysis
            scene_description = self._analyze_scene(cv_image, results)

            # Publish scene description
            if scene_description:
                scene_msg = SceneDescription()
                scene_msg.header = image_msg.header
                scene_msg.description = scene_description
                scene_msg.timestamp = image_msg.header.stamp
                self.safe_publish(self.scene_description_publisher, scene_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def _create_detection_message(self, detections, header):
        """Create ObjectDetectionArray message from detections"""
        from capstone_system_interfaces.msg import ObjectDetectionArray, ObjectDetection

        detection_array = ObjectDetectionArray()
        detection_array.header = header

        for detection in detections:
            obj_detection = ObjectDetection()
            obj_detection.object_class = detection['class']
            obj_detection.confidence = detection['confidence']
            obj_detection.bounding_box.x_offset = detection['bbox'][0]
            obj_detection.bounding_box.y_offset = detection['bbox'][1]
            obj_detection.bounding_box.width = detection['bbox'][2]
            obj_detection.bounding_box.height = detection['bbox'][3]

            # Add 3D position if available
            if 'position_3d' in detection:
                obj_detection.position_3d.x = detection['position_3d'][0]
                obj_detection.position_3d.y = detection['position_3d'][1]
                obj_detection.position_3d.z = detection['position_3d'][2]

            detection_array.detections.append(obj_detection)

        return detection_array

    def _analyze_scene(self, cv_image, detection_results):
        """Analyze scene and generate description"""
        # This would use more sophisticated analysis in real implementation
        # For now, return a mock description based on detected objects

        if 'detections' in detection_results:
            objects = [det['class'] for det in detection_results['detections']]
            if objects:
                return f"Scene contains: {', '.join(objects[:5])}"  # Limit to first 5 objects

        return "Scene analyzed - no notable objects detected"

    def analyze_scene_callback(self, request, response):
        """Analyze scene service callback"""
        try:
            # This would analyze the current scene
            # For now, we'll return a mock response
            response.success = True
            response.description = "Mock scene analysis - implement real analysis"
            response.object_count = 0
            response.analysis_timestamp = self.get_clock().now().to_msg()
        except Exception as e:
            self.get_logger().error(f'Error in analyze scene service: {e}')
            response.success = False
            response.message = str(e)

        return response

# Mock object detector for demonstration
class MockObjectDetector:
    def __init__(self):
        # In real implementation, this would be a real object detection model
        self.classes = ['person', 'chair', 'table', 'cup', 'bottle', 'keyboard', 'monitor']

    def detect(self, image, threshold=0.5):
        """Mock detection - in real implementation, this would use a real detector"""
        # Simulate detection results
        height, width = image.shape[:2]

        detections = []
        import random

        # Generate some mock detections
        for _ in range(random.randint(1, 5)):
            obj_class = random.choice(self.classes)
            conf = random.uniform(threshold, 1.0)

            # Random bounding box
            w = random.randint(50, 200)
            h = random.randint(50, 200)
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)

            detection = {
                'class': obj_class,
                'confidence': conf,
                'bbox': [x, y, w, h],
                'center': [x + w/2, y + h/2]
            }

            detections.append(detection)

        return detections
```

## 2. Communication Patterns Implementation

### 2.1 Service and Action Implementation

Create custom service and action interfaces for complex behaviors:

```python
# capstone_system_interfaces/srv/process_command.srv
string command
---
bool success
string message
capstone_system_interfaces/ParsedCommand parsed_command

# capstone_system_interfaces/srv/analyze_scene.srv
---
bool success
string message
string description
int32 object_count
builtin_interfaces/Time analysis_timestamp

# capstone_system_interfaces/msg/ObjectDetection.msg
string object_class
float32 confidence
# Bounding box in pixel coordinates
RegionOfInterest bounding_box
# 3D position in robot coordinate frame
geometry_msgs/Point position_3d
```

### 2.2 Safety Manager Node

Implement the safety manager that monitors all system activities:

```python
# safety_manager_node.py
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from capstone_system_interfaces.srv import ValidateAction
from capstone_system_interfaces.msg import SafetyAlert

class SafetyManagerNode(CapstoneBaseNode):
    def __init__(self):
        super().__init__('safety_manager_node')

        # Configuration parameters
        self.declare_parameter('enable_collision_detection', True)
        self.declare_parameter('enable_velocity_limits', True)
        self.declare_parameter('enable_proximity_safety', True)
        self.declare_parameter('safe_distance_threshold', 0.5)  # meters
        self.declare_parameter('max_linear_velocity', 0.5)  # m/s
        self.declare_parameter('max_angular_velocity', 1.0)  # rad/s

        # Publishers
        self.safety_alert_publisher = self.create_publisher(
            SafetyAlert, '/safety_alerts', self.qos_control
        )
        self.emergency_stop_publisher = self.create_publisher(
            Bool, '/emergency_stop', self.qos_control
        )

        # Subscribers to monitor system state
        self.robot_state_subscriber = self.create_subscription(
            String, '/robot_state', self.robot_state_callback, self.qos_control
        )
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, self.qos_control
        )
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, self.qos_sensor_data
        )

        # Services
        self.validate_action_service = self.create_service(
            ValidateAction, '/validate_action', self.validate_action_callback
        )

        # Safety state
        self.safety_enabled = True
        self.emergency_stop_active = False
        self.last_cmd_vel = None
        self.safety_violations = 0

        # Initialize safety parameters
        self.safe_distance_threshold = self.get_parameter('safe_distance_threshold').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value

        # Setup safety monitoring timer
        self.safety_timer = self.create_timer(0.1, self.safety_monitor_callback)  # 10Hz

        self.node_ready = True
        self.get_logger().info('Safety Manager Node initialized')

    def robot_state_callback(self, msg):
        """Monitor robot state for safety"""
        # Check for unsafe states
        if 'error' in msg.data.lower() or 'failure' in msg.data.lower():
            self.trigger_safety_alert(f'Detected error state: {msg.data}', 'critical')

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands for safety"""
        self.last_cmd_vel = msg

        # Check velocity limits
        if self.get_parameter('enable_velocity_limits').value:
            if abs(msg.linear.x) > self.max_linear_velocity:
                self.trigger_safety_alert(
                    f'Linear velocity limit exceeded: {msg.linear.x} > {self.max_linear_velocity}',
                    'warning'
                )

            if abs(msg.angular.z) > self.max_angular_velocity:
                self.trigger_safety_alert(
                    f'Angular velocity limit exceeded: {msg.angular.z} > {self.max_angular_velocity}',
                    'warning'
                )

    def scan_callback(self, msg):
        """Monitor sensor data for collision avoidance"""
        if not self.get_parameter('enable_collision_detection').value:
            return

        # Check for obstacles in path
        min_distance = min([r for r in msg.ranges if not np.isnan(r)], default=float('inf'))

        if min_distance < self.safe_distance_threshold:
            self.trigger_safety_alert(
                f'Obstacle detected at {min_distance:.2f}m, threshold is {self.safe_distance_threshold}m',
                'critical'
            )

    def validate_action_callback(self, request, response):
        """Validate action for safety before execution"""
        try:
            # Check various safety criteria
            is_safe = True
            reasons = []

            # Check velocity limits
            if hasattr(request, 'cmd_vel'):
                vel = request.cmd_vel
                if abs(vel.linear.x) > self.max_linear_velocity:
                    is_safe = False
                    reasons.append(f'Linear velocity too high: {vel.linear.x}')

                if abs(vel.angular.z) > self.max_angular_velocity:
                    is_safe = False
                    reasons.append(f'Angular velocity too high: {vel.angular.z}')

            # Check position constraints
            if hasattr(request, 'target_pose'):
                # Add position safety checks here
                pass

            response.is_safe = is_safe
            response.reasons = reasons
            response.validation_timestamp = self.get_clock().now().to_msg()

        except Exception as e:
            self.get_logger().error(f'Error in validate action service: {e}')
            response.is_safe = False
            response.reasons = [f'Validation error: {str(e)}']

        return response

    def safety_monitor_callback(self):
        """Periodic safety monitoring"""
        if not self.safety_enabled or self.emergency_stop_active:
            return

        # Additional safety checks can be added here
        # For example: check if robot is in restricted zones
        # Check for unusual sensor readings
        # Monitor system resource usage

        # If any safety violation is detected, trigger emergency stop
        if self.safety_violations > 10:  # Threshold for emergency stop
            self.emergency_stop()
            self.safety_violations = 0  # Reset counter

    def trigger_safety_alert(self, message: str, severity: str = 'warning'):
        """Trigger a safety alert"""
        alert_msg = SafetyAlert()
        alert_msg.header.stamp = self.get_clock().now().to_msg()
        alert_msg.message = message
        alert_msg.severity = severity
        alert_msg.timestamp = self.get_clock().now().to_msg()

        self.safe_publish(self.safety_alert_publisher, alert_msg)

        if severity == 'critical':
            self.safety_violations += 1
            self.get_logger().error(f'Safety Alert [{severity}]: {message}')
        else:
            self.get_logger().warn(f'Safety Alert [{severity}]: {message}')

    def emergency_stop(self):
        """Trigger emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            stop_msg = Bool()
            stop_msg.data = True
            self.safe_publish(self.emergency_stop_publisher, stop_msg)
            self.get_logger().fatal('EMERGENCY STOP ACTIVATED')

    def reset_safety(self):
        """Reset safety system"""
        self.emergency_stop_active = False
        self.safety_violations = 0
        self.get_logger().info('Safety system reset')
```

## 3. Hands-on Exercises

### Exercise 1: Core Node Implementation
**Objective:** Implement the core ROS 2 nodes for the capstone system.

**Prerequisites:**
- Completion of Phase 1 (System Architecture)
- Understanding of ROS 2 concepts
- Python programming experience

**Steps:**
1. Create the CapstoneBaseNode base class with common functionality
2. Implement the VoiceRecognitionNode with speech recognition capabilities
3. Implement the LanguageProcessorNode with LLM integration
4. Implement the VisionSystemNode with image processing
5. Test individual node functionality
6. Validate message passing and communication patterns

**Expected Outcome:** Working ROS 2 nodes that implement the communication framework components.

**Troubleshooting Tips:**
- Ensure proper QoS profile selection for different message types
- Verify parameter declarations and configurations
- Check message type compatibility between nodes
- Monitor resource usage during operation

### Exercise 2: Service and Action Implementation
**Objective:** Implement service and action interfaces for complex behaviors.

**Prerequisites:**
- Completed Exercise 1
- Understanding of ROS 2 services and actions
- Experience with custom message definitions

**Steps:**
1. Define custom service and message interfaces
2. Implement service callbacks for command processing
3. Create action servers for complex task execution
4. Test service and action communication
5. Validate error handling and timeouts
6. Document the service and action interfaces

**Expected Outcome:** Functional service and action interfaces that enable complex behaviors.

### Exercise 3: Safety Manager Implementation
**Objective:** Implement the safety manager that monitors system activities.

**Prerequisites:**
- Completed previous exercises
- Understanding of safety concepts in robotics
- Experience with monitoring and validation systems

**Steps:**
1. Implement the SafetyManagerNode with monitoring capabilities
2. Create safety validation services
3. Implement emergency stop functionality
4. Test safety monitoring with simulated violations
5. Validate safety system independence
6. Document safety procedures and protocols

**Expected Outcome:** Comprehensive safety manager that monitors and protects the system.

## 4. Safety and Ethical Considerations

When implementing the ROS 2 communication framework:
- Ensure safety-critical communications are reliable and timely
- Implement proper error handling and fallback behaviors
- Consider privacy implications of data transmission
- Plan for graceful degradation when communication fails
- Maintain human oversight for critical decisions
- Implement secure communication protocols
- Ensure safety systems operate independently of other systems

## 5. Phase Summary

In this phase, you've completed:
- Implementation of the ROS 2 communication framework for the capstone system
- Creation of core nodes for voice recognition, language processing, and vision
- Establishment of communication patterns and message passing
- Implementation of service and action interfaces for complex behaviors
- Creation of a safety manager for system monitoring
- Validation and testing of the communication framework

The ROS 2 framework you've implemented provides the essential communication infrastructure that enables all system components to work together effectively.

## 6. Assessment Questions

### Multiple Choice
1. What is the primary purpose of the SafetyManagerNode in the communication framework?
   a) To process voice commands
   b) To manage all system communications
   c) To monitor system activities and ensure safety
   d) To handle vision processing

   Answer: c) To monitor system activities and ensure safety

2. Which QoS profile is most appropriate for sensor data in the communication framework?
   a) RELIABLE with KEEP_ALL history
   b) BEST_EFFORT with KEEP_LAST history
   c) RELIABLE with KEEP_LAST history
   d) BEST_EFFORT with KEEP_ALL history

   Answer: b) BEST_EFFORT with KEEP_LAST history

### Practical Questions
1. Implement a complete ROS 2 communication framework that includes voice recognition, language processing, vision processing, and safety management nodes with proper message passing and service interfaces.

## 7. Next Steps

After completing Phase 2, you should:
- Test the communication framework thoroughly with various scenarios
- Validate performance and reliability of the ROS 2 nodes
- Ensure all safety mechanisms are functioning properly
- Prepare for Phase 3: Digital Twin and Simulation Environment Integration
- Review and optimize the communication patterns for performance
- Document the ROS 2 framework for future reference

The communication framework you've implemented serves as the backbone for the entire capstone system, so ensure it is robust, reliable, and safe before proceeding to the next phase.