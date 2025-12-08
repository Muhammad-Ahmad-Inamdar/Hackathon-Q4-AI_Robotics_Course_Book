---
sidebar_position: 5
learning_objectives:
  - Integrate AI-Robot Brain components with the capstone system
  - Implement Vision-Language-Action (VLA) capabilities for autonomous behavior
  - Create multimodal AI systems that combine vision, language, and action
  - Integrate Isaac Sim and Isaac ROS for advanced AI capabilities
  - Implement cognitive planning and reasoning systems
  - Validate AI system performance and safety
prerequisites:
  - Completion of Phases 1-3 (System Architecture, ROS 2 Framework, Simulation Environment)
  - Understanding of AI and machine learning concepts
  - Experience with Isaac Sim and Isaac ROS (Module 3)
  - Access to appropriate hardware with GPU support for AI processing
estimated_time: "15 hours"
---

# Phase 4: AI-Robot Brain and VLA Integration

## Learning Objectives

By completing this phase, you will be able to:
- Integrate AI-Robot Brain components with the capstone system
- Implement Vision-Language-Action (VLA) capabilities for autonomous behavior
- Create multimodal AI systems that combine vision, language, and action
- Integrate Isaac Sim and Isaac ROS for advanced perception and AI capabilities
- Implement cognitive planning and reasoning systems using LLMs
- Validate AI system performance, safety, and ethical considerations
- Ensure the AI components meet real-time performance requirements
- Establish safety protocols for AI-driven robotic behavior

## Introduction

Phase 4 focuses on integrating the AI-Robot Brain and Vision-Language-Action (VLA) capabilities into your capstone system. This phase brings together the advanced AI concepts from Modules 3 and 4, implementing sophisticated perception, reasoning, and action capabilities that enable your humanoid robot to understand natural language commands, perceive its environment visually, and execute complex tasks autonomously.

The AI-Robot Brain integration is crucial because it provides:
- Cognitive capabilities for understanding and reasoning
- Multimodal perception combining vision and language
- Autonomous decision-making and planning
- Adaptive learning and improvement
- Natural human-robot interaction through VLA capabilities
- Safety-aware AI-driven behavior

## 1. AI-Robot Brain Integration

### 1.1 Perception System Implementation

Implement the perception system using Isaac ROS components:

```python
# ai_perception_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String, Header
from capstone_system_interfaces.msg import SceneUnderstanding, ObjectInteraction
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import List, Dict, Any, Optional
import threading
import queue
import time

class AI Perception Node(Node):
    def __init__(self):
        super().__init__('ai_perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('enable_object_detection', True)
        self.declare_parameter('enable_segmentation', False)
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('max_detection_range', 5.0)
        self.declare_parameter('enable_tracking', True)

        # Publishers
        self.scene_understanding_publisher = self.create_publisher(
            SceneUnderstanding, '/ai/scene_understanding', 10
        )
        self.object_interactions_publisher = self.create_publisher(
            ObjectInteraction, '/ai/object_interactions', 10
        )
        self.debug_image_publisher = self.create_publisher(
            Image, '/ai/debug_image', 10
        )

        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2, '/depth/points', self.pointcloud_callback, 10
        )

        # Isaac ROS integration components
        self.isaac_ai_initialized = False
        self.camera_intrinsics = None
        self.pointcloud_data = None

        # Processing queues and threads
        self.image_queue = queue.Queue(maxsize=2)
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # Object tracking
        self.tracked_objects = {}
        self.next_track_id = 0

        self.node_ready = True
        self.get_logger().info('AI Perception Node initialized')

    def camera_info_callback(self, msg):
        """Process camera intrinsics for 3D reconstruction"""
        self.camera_intrinsics = {
            'fx': msg.k[0],  # Focal length x
            'fy': msg.k[4],  # Focal length y
            'cx': msg.k[2],  # Principal point x
            'cy': msg.k[5],  # Principal point y
            'width': msg.width,
            'height': msg.height
        }

    def pointcloud_callback(self, msg):
        """Process point cloud data for 3D object understanding"""
        # Store point cloud for 3D object association
        self.pointcloud_data = msg

    def image_callback(self, msg):
        """Process incoming camera images for AI perception"""
        try:
            # Add image to processing queue (non-blocking)
            try:
                self.image_queue.put_nowait(msg)
            except queue.Full:
                # Drop oldest image if queue is full
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put_nowait(msg)
                except queue.Empty:
                    pass  # Queue is empty, just add the new one
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def processing_worker(self):
        """Worker thread for AI perception processing"""
        while rclpy.ok():
            try:
                # Get image from queue with timeout
                image_msg = self.image_queue.get(timeout=1.0)

                # Process image with AI perception
                self.process_image_ai(image_msg)

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.start_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    self.frame_count = 0
                    self.start_time = current_time
                    self.get_logger().debug(f'AI Perception FPS: {self.fps:.2f}')

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Error in perception worker: {e}')

    def process_image_ai(self, image_msg):
        """Process image using AI perception techniques"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Perform AI-based perception
            perception_results = self.ai_perception_pipeline(cv_image)

            # Create and publish scene understanding message
            scene_msg = self.create_scene_understanding_message(
                perception_results, image_msg.header
            )

            if scene_msg:
                self.scene_understanding_publisher.publish(scene_msg)

            # Create and publish object interaction message
            interaction_msg = self.create_object_interaction_message(
                perception_results, image_msg.header
            )

            if interaction_msg:
                self.object_interactions_publisher.publish(interaction_msg)

            # Publish debug image if needed
            if self.get_parameter('publish_debug_image').value:
                debug_image = self.draw_perception_results(cv_image, perception_results)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                debug_msg.header = image_msg.header
                self.debug_image_publisher.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in AI perception processing: {e}')

    def ai_perception_pipeline(self, cv_image):
        """AI perception pipeline combining multiple techniques"""
        results = {
            'objects': [],
            'relationships': [],
            'scene_description': '',
            'actionable_items': [],
            'safety_considerations': []
        }

        # Object detection (using Isaac ROS equivalent or mock implementation)
        objects = self.detect_objects(cv_image)
        results['objects'] = objects

        # Scene understanding
        scene_desc = self.understand_scene(cv_image, objects)
        results['scene_description'] = scene_desc

        # Spatial relationships
        relationships = self.analyze_spatial_relationships(objects)
        results['relationships'] = relationships

        # Actionable items identification
        actionable = self.identify_actionable_items(objects)
        results['actionable_items'] = actionable

        # Safety considerations
        safety_considerations = self.analyze_safety_considerations(objects)
        results['safety_considerations'] = safety_considerations

        # Object tracking
        if self.get_parameter('enable_tracking').value:
            self.update_object_tracking(objects)

        return results

    def detect_objects(self, cv_image):
        """Perform object detection using AI model"""
        # In a real implementation, this would use Isaac ROS DNN Inference
        # For this example, we'll use a mock implementation
        height, width = cv_image.shape[:2]

        # Mock object detections for demonstration
        import random

        objects = []
        for i in range(random.randint(2, 5)):  # 2-5 random objects
            obj_class = random.choice(['person', 'chair', 'table', 'cup', 'bottle', 'box'])
            confidence = random.uniform(0.6, 0.95)

            # Random bounding box
            bbox_width = random.randint(50, 200)
            bbox_height = random.randint(50, 200)
            x = random.randint(0, width - bbox_width)
            y = random.randint(0, height - bbox_height)

            # Calculate center and 3D position if camera intrinsics available
            center_x = x + bbox_width / 2
            center_y = y + bbox_height / 2

            # Estimate 3D position using mock depth
            depth_estimate = random.uniform(0.5, 3.0)  # meters

            # Calculate 3D position (simplified)
            if self.camera_intrinsics:
                fx = self.camera_intrinsics['fx']
                fy = self.camera_intrinsics['fy']
                cx = self.camera_intrinsics['cx']
                cy = self.camera_intrinsics['cy']

                # Convert pixel coordinates to 3D using camera intrinsics
                pos_x = (center_x - cx) * depth_estimate / fx
                pos_y = (center_y - cy) * depth_estimate / fy
                pos_z = depth_estimate
            else:
                # Use mock coordinates
                pos_x = (center_x - width/2) * 0.01
                pos_y = (center_y - height/2) * 0.01
                pos_z = depth_estimate

            object_info = {
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x, y, x + bbox_width, y + bbox_height],
                'center_2d': [center_x, center_y],
                'position_3d': [pos_x, pos_y, pos_z],
                'size_3d': [bbox_width * 0.01, bbox_height * 0.01, depth_estimate * 0.1],  # mock size
                'track_id': self.assign_track_id([x, y, x + bbox_width, y + bbox_height])
            }

            objects.append(object_info)

        return objects

    def understand_scene(self, cv_image, objects):
        """Generate natural language description of the scene"""
        # This would use more sophisticated scene understanding in a real implementation
        if not objects:
            return "Scene appears empty or no objects detected"

        # Count objects by class
        class_counts = {}
        for obj in objects:
            class_name = obj['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Generate description
        description_parts = []
        for class_name, count in class_counts.items():
            if count == 1:
                description_parts.append(f"a {class_name}")
            else:
                description_parts.append(f"{count} {class_name}s")

        if len(description_parts) == 1:
            scene_desc = f"Scene contains {description_parts[0]}"
        else:
            scene_desc = f"Scene contains: {', '.join(description_parts[:-1])} and {description_parts[-1]}"

        return scene_desc

    def analyze_spatial_relationships(self, objects):
        """Analyze spatial relationships between objects"""
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Calculate distance between object centers
                    center1 = obj1['center_2d']
                    center2 = obj2['center_2d']

                    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

                    # Determine spatial relationship based on position
                    dx = center2[0] - center1[0]
                    dy = center2[1] - center1[1]

                    if abs(dx) > abs(dy):  # Horizontal relationship dominates
                        if dx > 0:
                            relationship = f"{obj2['class']} is to the right of {obj1['class']}"
                        else:
                            relationship = f"{obj2['class']} is to the left of {obj1['class']}"
                    else:  # Vertical relationship dominates
                        if dy > 0:
                            relationship = f"{obj2['class']} is below {obj1['class']}"
                        else:
                            relationship = f"{obj2['class']} is above {obj1['class']}"

                    relationships.append({
                        'object1': obj1['class'],
                        'object2': obj2['class'],
                        'relationship': relationship,
                        'distance_pixels': distance
                    })

        return relationships

    def identify_actionable_items(self, objects):
        """Identify objects that can be interacted with"""
        actionable_classes = ['cup', 'bottle', 'box', 'chair', 'person']
        actionable_items = []

        for obj in objects:
            if obj['class'] in actionable_classes and obj['confidence'] > 0.7:
                # Determine if object is reachable based on 3D position
                pos_3d = obj['position_3d']
                distance_3d = np.sqrt(pos_3d[0]**2 + pos_3d[1]**2 + pos_3d[2]**2)

                if distance_3d <= self.get_parameter('max_detection_range').value:
                    actionable_items.append({
                        'class': obj['class'],
                        'position_3d': pos_3d,
                        'confidence': obj['confidence'],
                        'bbox': obj['bbox'],
                        'reachable': distance_3d <= 1.5,  # 1.5m reach threshold
                        'interaction_type': self.determine_interaction_type(obj['class'])
                    })

        return actionable_items

    def determine_interaction_type(self, object_class):
        """Determine appropriate interaction type for an object"""
        interaction_map = {
            'cup': 'grasp',
            'bottle': 'grasp',
            'box': 'grasp',
            'chair': 'navigate_to',
            'person': 'greet'
        }
        return interaction_map.get(object_class, 'inspect')

    def analyze_safety_considerations(self, objects):
        """Analyze safety considerations for detected objects"""
        safety_considerations = []

        for obj in objects:
            if obj['class'] in ['person', 'human']:
                if obj['confidence'] > 0.8:
                    safety_considerations.append({
                        'type': 'human_presence',
                        'description': f'Person detected at {obj["position_3d"]}',
                        'priority': 'high'
                    })
            elif obj['class'] in ['bottle', 'cup']:
                # Check if object is in pathway
                if abs(obj['position_3d'][0]) < 0.5 and obj['position_3d'][1] < 1.0:  # In front and close
                    safety_considerations.append({
                        'type': 'obstacle',
                        'description': f'{obj["class"]} in path at {obj["position_3d"]}',
                        'priority': 'medium'
                    })

        return safety_considerations

    def assign_track_id(self, bbox):
        """Assign or update object tracking ID"""
        # Simple IoU-based tracking for demonstration
        bbox_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Look for existing track that matches this detection
        best_match_id = None
        best_iou = 0

        for track_id, track_info in self.tracked_objects.items():
            track_bbox = track_info['bbox']
            track_center = [(track_bbox[0] + track_bbox[2])/2, (track_bbox[1] + track_bbox[3])/2]

            # Calculate distance between centers
            center_dist = np.sqrt((bbox_center[0] - track_center[0])**2 + (bbox_center[1] - track_center[1])**2)

            # Calculate IoU
            x1 = max(bbox[0], track_bbox[0])
            y1 = max(bbox[1], track_bbox[1])
            x2 = min(bbox[2], track_bbox[2])
            y2 = min(bbox[3], track_bbox[3])

            if x1 < x2 and y1 < y2:
                intersection_area = (x2 - x1) * (y2 - y1)
                union_area = bbox_area + (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1]) - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0

                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_match_id = track_id

        if best_match_id is not None:
            # Update existing track
            self.tracked_objects[best_match_id]['bbox'] = bbox
            self.tracked_objects[best_match_id]['last_seen'] = time.time()
            return best_match_id
        else:
            # Create new track
            new_id = self.next_track_id
            self.tracked_objects[new_id] = {
                'bbox': bbox,
                'last_seen': time.time(),
                'class': None  # Will be set when associated with detection
            }
            self.next_track_id = (self.next_track_id + 1) % 1000  # Cycle IDs to prevent overflow
            return new_id

    def update_object_tracking(self, objects):
        """Update object tracking with new detections"""
        current_time = time.time()

        # Associate detections with tracks
        for obj in objects:
            # The assign_track_id function already handles association
            # Here we just clean up old tracks
            pass

        # Remove old tracks that haven't been seen recently
        tracks_to_remove = []
        for track_id, track_info in self.tracked_objects.items():
            if current_time - track_info['last_seen'] > 5.0:  # Remove tracks not seen in 5 seconds
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]

    def create_scene_understanding_message(self, perception_results, header):
        """Create SceneUnderstanding message from perception results"""
        if not perception_results['objects']:
            return None

        scene_msg = SceneUnderstanding()
        scene_msg.header = header
        scene_msg.timestamp = self.get_clock().now().to_msg()
        scene_msg.scene_description = perception_results['scene_description']

        # Populate objects
        for obj in perception_results['objects']:
            object_msg = ObjectHypothesisWithPose()
            object_msg.hypothesis.class_id = obj['class']
            object_msg.hypothesis.score = obj['confidence']

            # Set 3D pose
            pose = Pose()
            pose.position.x = float(obj['position_3d'][0])
            pose.position.y = float(obj['position_3d'][1])
            pose.position.z = float(obj['position_3d'][2])
            # Simple orientation assumption
            pose.orientation.w = 1.0

            object_msg.pose.pose = pose
            scene_msg.objects.append(object_msg)

        # Populate relationships
        for rel in perception_results['relationships']:
            # Add relationship information to message
            pass

        # Populate safety considerations
        for safety_item in perception_results['safety_considerations']:
            # Add safety information to message
            pass

        return scene_msg

    def create_object_interaction_message(self, perception_results, header):
        """Create ObjectInteraction message from perception results"""
        actionable_items = perception_results['actionable_items']
        if not actionable_items:
            return None

        interaction_msg = ObjectInteraction()
        interaction_msg.header = header
        interaction_msg.timestamp = self.get_clock().now().to_msg()

        for item in actionable_items:
            # Create interaction suggestion
            interaction = ObjectInteraction.Interaction()
            interaction.object_class = item['class']
            interaction.interaction_type = item['interaction_type']
            interaction.position.x = item['position_3d'][0]
            interaction.position.y = item['position_3d'][1]
            interaction.position.z = item['position_3d'][2]
            interaction.confidence = item['confidence']
            interaction.is_reachable = item['reachable']

            interaction_msg.interactions.append(interaction)

        return interaction_msg

    def draw_perception_results(self, cv_image, perception_results):
        """Draw perception results on image for debugging"""
        debug_image = cv_image.copy()

        # Draw object detections
        for obj in perception_results['objects']:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Label with class and confidence
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            if 'track_id' in obj:
                label += f" [ID:{obj['track_id']}]"

            cv2.putText(debug_image, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw spatial relationships
        for rel in perception_results['relationships'][:3]:  # Limit to first 3 for clarity
            # In a real implementation, you might draw arrows or lines
            # between related objects
            pass

        # Add FPS information
        cv2.putText(debug_image, f"AI Perception FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return debug_image
```

### 1.2 Cognitive Planning Node

Implement the cognitive planning system using LLM integration:

```python
# cognitive_planning_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from capstone_system_interfaces.msg import ParsedCommand, TaskPlan, SceneUnderstanding
from capstone_system_interfaces.srv import PlanTask, ValidateAction
import openai
import json
import time
from typing import Dict, List, Any, Optional
import threading
import queue

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Declare parameters
        self.declare_parameter('openai_api_key', '', ParameterDescriptor(description='OpenAI API key'))
        self.declare_parameter('llm_model', 'gpt-3.5-turbo', ParameterDescriptor(description='LLM model to use'))
        self.declare_parameter('temperature', 0.3, ParameterDescriptor(description='LLM temperature'))
        self.declare_parameter('max_tokens', 1000, ParameterDescriptor(description='Maximum tokens for LLM'))
        self.declare_parameter('enable_context_history', True, ParameterDescriptor(description='Enable conversation context'))
        self.declare_parameter('context_window_size', 10, ParameterDescriptor(description='Size of context window'))

        # Initialize OpenAI
        api_key = self.get_parameter('openai_api_key').value
        if api_key:
            openai.api_key = api_key
            self.llm_model = self.get_parameter('llm_model').value
            self.temperature = self.get_parameter('temperature').value
            self.max_tokens = self.get_parameter('max_tokens').value
            self.llm_initialized = True
            self.get_logger().info('LLM client initialized for cognitive planning')
        else:
            self.get_logger().warn('No OpenAI API key provided, LLM functionality disabled')
            self.llm_initialized = False

        # Publishers
        self.task_plan_publisher = self.create_publisher(
            TaskPlan, '/ai/task_plans', 10
        )
        self.system_response_publisher = self.create_publisher(
            String, '/system/responses', 10
        )

        # Subscribers
        self.parsed_command_subscriber = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.parsed_command_callback, 10
        )
        self.scene_understanding_subscriber = self.create_subscription(
            SceneUnderstanding, '/ai/scene_understanding', self.scene_understanding_callback, 10
        )

        # Services
        self.plan_task_service = self.create_service(
            PlanTask, '/plan_task', self.plan_task_callback
        )
        self.validate_action_service = self.create_service(
            ValidateAction, '/validate_action', self.validate_action_callback
        )

        # Internal state
        self.current_scene = None
        self.conversation_history = []
        self.context_window_size = self.get_parameter('context_window_size').value
        self.planning_queue = queue.Queue()
        self.planning_thread = threading.Thread(target=self.planning_worker, daemon=True)
        self.planning_thread.start()

        # Safety validators
        self.safety_validator = SafetyValidator(self)

        self.node_ready = True
        self.get_logger().info('Cognitive Planning Node initialized')

    def scene_understanding_callback(self, msg):
        """Update current scene understanding"""
        self.current_scene = {
            'timestamp': msg.header.stamp,
            'description': msg.scene_description,
            'objects': [{'class': obj.hypothesis.class_id, 'confidence': obj.hypothesis.score}
                       for obj in msg.objects],
            'relationships': msg.relationships  # Assuming this exists in the message
        }

    def parsed_command_callback(self, msg):
        """Process parsed commands and generate task plans"""
        if not self.llm_initialized:
            self.get_logger().warn('LLM not initialized, cannot process command')
            return

        try:
            # Create a planning request
            planning_request = {
                'command': msg.original_command,
                'intent': msg.intent,
                'action': msg.action,
                'objects': msg.objects,
                'locations': msg.locations,
                'current_scene': self.current_scene,
                'context_history': self.conversation_history[-self.context_window_size:]
            }

            # Add to planning queue
            self.planning_queue.put(planning_request)

        except Exception as e:
            self.get_logger().error(f'Error processing parsed command: {e}')

    def planning_worker(self):
        """Worker thread for LLM-based planning"""
        while rclpy.ok():
            try:
                # Get planning request from queue
                request = self.planning_queue.get(timeout=1.0)

                # Generate task plan using LLM
                task_plan = self.generate_task_plan_llm(request)

                if task_plan:
                    # Publish the task plan
                    self.task_plan_publisher.publish(task_plan)

                    # Generate system response
                    response = self.generate_system_response(request, task_plan)
                    if response:
                        response_msg = String()
                        response_msg.data = response
                        self.system_response_publisher.publish(response_msg)

                # Update conversation history
                if self.get_parameter('enable_context_history').value:
                    self.conversation_history.append({
                        'command': request['command'],
                        'plan_generated': task_plan is not None,
                        'timestamp': time.time()
                    })

                    # Limit history size
                    if len(self.conversation_history) > self.context_window_size:
                        self.conversation_history = self.conversation_history[-self.context_window_size:]

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Error in planning worker: {e}')

    def generate_task_plan_llm(self, request: Dict[str, Any]) -> Optional[TaskPlan]:
        """Generate task plan using LLM"""
        try:
            # Create a comprehensive prompt for task planning
            prompt = self.create_planning_prompt(request)

            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.get_planning_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            plan_data = self.extract_json_from_response(response_text)

            if plan_data:
                return self.create_task_plan_message(plan_data, request)
            else:
                self.get_logger().error(f'Could not extract valid JSON from LLM response: {response_text}')
                return None

        except Exception as e:
            self.get_logger().error(f'Error in LLM task planning: {e}')
            return None

    def create_planning_prompt(self, request: Dict[str, Any]) -> str:
        """Create prompt for LLM-based task planning"""
        prompt = f"""
        You are an advanced AI planning system for a humanoid robot. Generate a detailed task plan based on the user command and current scene understanding.

        USER COMMAND: "{request['command']}"
        INTENT: {request['intent']}
        ACTION: {request['action']}
        OBJECTS: {request['objects']}
        LOCATIONS: {request['locations']}

        CURRENT SCENE: {request['current_scene']['description'] if request['current_scene'] else 'No scene data available'}

        OBJECTS IN SCENE: {[obj['class'] for obj in request['current_scene']['objects']] if request['current_scene'] else []}

        CONTEXT HISTORY: {request['context_history'] if request['context_history'] else []}

        Generate a detailed task plan in JSON format with the following structure:
        {{
            "plan_id": "unique_plan_identifier",
            "description": "brief description of the plan",
            "safety_considerations": ["consideration1", "consideration2"],
            "tasks": [
                {{
                    "task_id": "unique_task_id",
                    "description": "what to do",
                    "action_type": "navigation|manipulation|perception|communication|wait",
                    "parameters": {{
                        "target_pose": {{"x": float, "y": float, "z": float, "qx": float, "qy": float, "qz": float, "qw": float}},
                        "target_object": "object_name",
                        "action_parameters": {{"param1": "value1"}}
                    }},
                    "dependencies": ["task_id_1", "task_id_2"],
                    "priority": integer (higher number = higher priority),
                    "estimated_duration": float (seconds),
                    "success_criteria": "how to determine if task was successful"
                }}
            ],
            "estimated_total_duration": float (seconds),
            "confidence": float (0.0 to 1.0)
        }}

        The plan should be:
        1. Safe - consider safety constraints and potential hazards
        2. Feasible - based on robot capabilities and scene understanding
        3. Efficient - minimize unnecessary movements
        4. Specific - include concrete parameters for each action
        5. Sequential - respect dependencies between tasks

        Respond ONLY with the JSON plan, no additional text.
        """

        return prompt

    def get_planning_system_prompt(self) -> str:
        """Get system prompt for planning"""
        return """
        You are an AI planning system for a humanoid robot. Your task is to generate detailed, executable task plans based on natural language commands and scene understanding. Be precise, consider safety, and ensure plans are executable by a robot with navigation and manipulation capabilities. Always respond in valid JSON format only.
        """

    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        try:
            # Look for JSON within code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON structure
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return None

    def create_task_plan_message(self, plan_data: Dict[str, Any], original_request: Dict[str, Any]) -> TaskPlan:
        """Create TaskPlan ROS message from JSON data"""
        plan_msg = TaskPlan()
        plan_msg.header.stamp = self.get_clock().now().to_msg()
        plan_msg.header.frame_id = "map"
        plan_msg.plan_id = plan_data.get('plan_id', 'unknown_plan')
        plan_msg.description = plan_data.get('description', 'No description')
        plan_msg.safety_considerations = plan_data.get('safety_considerations', [])
        plan_msg.estimated_total_duration = plan_data.get('estimated_total_duration', 0.0)
        plan_msg.confidence = plan_data.get('confidence', 0.0)
        plan_msg.original_command = original_request['command']

        # Create task messages
        for task_data in plan_data.get('tasks', []):
            task_msg = TaskPlan.Task()
            task_msg.task_id = task_data.get('task_id', 'unknown_task')
            task_msg.description = task_data.get('description', '')
            task_msg.action_type = task_data.get('action_type', 'unknown')
            task_msg.priority = task_data.get('priority', 0)
            task_msg.estimated_duration = task_data.get('estimated_duration', 0.0)
            task_msg.success_criteria = task_data.get('success_criteria', '')

            # Set parameters
            params = task_data.get('parameters', {})
            if 'target_pose' in params:
                pose_data = params['target_pose']
                pose = PoseStamped()
                pose.pose.position.x = pose_data.get('x', 0.0)
                pose.pose.position.y = pose_data.get('y', 0.0)
                pose.pose.position.z = pose_data.get('z', 0.0)
                pose.pose.orientation.x = pose_data.get('qx', 0.0)
                pose.pose.orientation.y = pose_data.get('qy', 0.0)
                pose.pose.orientation.z = pose_data.get('qz', 0.0)
                pose.pose.orientation.w = pose_data.get('qw', 1.0)
                task_msg.target_pose = pose

            task_msg.target_object = params.get('target_object', '')

            # Action parameters
            action_params = params.get('action_parameters', {})
            task_msg.action_parameters = json.dumps(action_params)

            # Dependencies
            task_msg.dependencies = task_data.get('dependencies', [])

            plan_msg.tasks.append(task_msg)

        return plan_msg

    def generate_system_response(self, request: Dict[str, Any], task_plan: TaskPlan) -> str:
        """Generate natural language response to the user"""
        try:
            prompt = f"""
            Generate a natural, friendly response to the user based on their command and the plan you've generated.

            USER COMMAND: "{request['command']}"
            GENERATED PLAN: {json.dumps({
                'description': task_plan.description,
                'tasks': [task.description for task in task_plan.tasks]
            })}

            Current scene: {request['current_scene']['description'] if request['current_scene'] else 'unknown'}

            Generate a response that acknowledges the command, explains what you're going to do, and mentions any relevant scene information. Keep it concise but informative.
            """

            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant. Respond naturally and informatively to user commands."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.get_logger().error(f'Error generating system response: {e}')
            return f"I understand your command '{request['command']}'. I'm working on that for you."

    def plan_task_callback(self, request, response):
        """Service callback for manual task planning"""
        if not self.llm_initialized:
            response.success = False
            response.message = 'LLM not initialized'
            return response

        try:
            # Create planning request from service call
            planning_request = {
                'command': request.command,
                'intent': request.intent if hasattr(request, 'intent') else '',
                'action': request.action if hasattr(request, 'action') else '',
                'objects': request.objects if hasattr(request, 'objects') else [],
                'locations': request.locations if hasattr(request, 'locations') else [],
                'current_scene': self.current_scene,
                'context_history': self.conversation_history[-self.context_window_size:]
            }

            # Generate plan synchronously for service response
            task_plan = self.generate_task_plan_llm(planning_request)

            if task_plan:
                response.success = True
                response.message = 'Plan generated successfully'
                response.task_plan = task_plan
            else:
                response.success = False
                response.message = 'Failed to generate plan'

        except Exception as e:
            self.get_logger().error(f'Error in plan task service: {e}')
            response.success = False
            response.message = f'Error generating plan: {str(e)}'

        return response

    def validate_action_callback(self, request, response):
        """Service callback for action validation"""
        try:
            # Validate the proposed action for safety and feasibility
            is_safe, safety_issues = self.safety_validator.validate_action(request.action)

            response.is_safe = is_safe
            response.safety_issues = safety_issues
            response.confidence = 0.95 if is_safe else 0.1

            if is_safe:
                response.message = 'Action is safe to execute'
            else:
                response.message = f'Action has safety concerns: {", ".join(safety_issues)}'

        except Exception as e:
            self.get_logger().error(f'Error in validate action service: {e}')
            response.is_safe = False
            response.safety_issues = [f'Validation error: {str(e)}']
            response.message = f'Validation error: {str(e)}'
            response.confidence = 0.0

        return response

class SafetyValidator:
    """Safety validation system for cognitive planning"""

    def __init__(self, node):
        self.node = node
        self.safety_rules = {
            'navigation': self.validate_navigation_safety,
            'manipulation': self.validate_manipulation_safety,
            'perception': self.validate_perception_safety,
            'communication': self.validate_communication_safety
        }

    def validate_action(self, action) -> tuple[bool, List[str]]:
        """Validate action for safety"""
        issues = []

        # Check if action type is supported
        action_type = getattr(action, 'action_type', 'unknown')
        if action_type in self.safety_rules:
            is_safe, action_issues = self.safety_rules[action_type](action)
            issues.extend(action_issues)
        else:
            issues.append(f'Unknown action type: {action_type}')

        # Check general safety rules
        general_issues = self.check_general_safety(action)
        issues.extend(general_issues)

        return len(issues) == 0, issues

    def validate_navigation_safety(self, action) -> tuple[bool, List[str]]:
        """Validate navigation action safety"""
        issues = []

        # Check target position is safe
        if hasattr(action, 'target_pose'):
            pose = action.target_pose.pose
            x, y, z = pose.position.x, pose.position.y, pose.position.z

            # Check for extreme positions
            if abs(x) > 10.0 or abs(y) > 10.0:
                issues.append(f'Navigation target too far: ({x:.2f}, {y:.2f})')

            if z < -1.0:
                issues.append(f'Invalid Z position: {z:.2f}')

        return len(issues) == 0, issues

    def validate_manipulation_safety(self, action) -> tuple[bool, List[str]]:
        """Validate manipulation action safety"""
        issues = []

        # Check if target object is safe to manipulate
        if hasattr(action, 'target_object'):
            obj_class = action.target_object

            # Dangerous objects
            dangerous_objects = ['knife', 'scissors', 'glass', 'chemical']
            if obj_class.lower() in dangerous_objects:
                issues.append(f'Dangerous object detected: {obj_class}')

        return len(issues) == 0, issues

    def validate_perception_safety(self, action) -> tuple[bool, List[str]]:
        """Validate perception action safety"""
        issues = []

        # Perception actions are generally safe
        return True, issues

    def validate_communication_safety(self, action) -> tuple[bool, List[str]]:
        """Validate communication action safety"""
        issues = []

        # Check for inappropriate content
        if hasattr(action, 'parameters') and isinstance(action.parameters, str):
            content = action.parameters.lower()
            inappropriate_words = ['harm', 'danger', 'unsafe', 'inappropriate']
            for word in inappropriate_words:
                if word in content:
                    issues.append(f'Potentially inappropriate content detected: {word}')

        return len(issues) == 0, issues

    def check_general_safety(self, action) -> List[str]:
        """Check general safety rules"""
        issues = []

        # Check if safety systems are operational
        # This would integrate with the safety manager from other modules

        return issues
```

## 2. Vision-Language-Action Integration

### 2.1 VLA Coordinator Node

Create the main coordinator that integrates vision, language, and action:

```python
# vla_coordinator_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped
from capstone_system_interfaces.msg import ParsedCommand, TaskPlan, SceneUnderstanding, ObjectInteraction
from capstone_system_interfaces.srv import ExecuteTask
import threading
import time
from typing import Dict, List, Any, Optional
import json

class VLA_CoordinatorNode(Node):
    def __init__(self):
        super().__init__('vla_coordinator_node')

        # Declare parameters
        self.declare_parameter('enable_vla_coordination', True)
        self.declare_parameter('task_execution_timeout', 30.0)
        self.declare_parameter('coordination_frequency', 10.0)

        # Publishers
        self.action_command_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        self.status_publisher = self.create_publisher(
            String, '/vla/status', 10
        )

        # Subscribers
        self.parsed_command_subscriber = self.create_subscription(
            ParsedCommand, '/parsed_commands', self.parsed_command_callback, 10
        )
        self.task_plan_subscriber = self.create_subscription(
            TaskPlan, '/ai/task_plans', self.task_plan_callback, 10
        )
        self.scene_understanding_subscriber = self.create_subscription(
            SceneUnderstanding, '/ai/scene_understanding', self.scene_callback, 10
        )
        self.object_interaction_subscriber = self.create_subscription(
            ObjectInteraction, '/ai/object_interactions', self.object_interaction_callback, 10
        )

        # Services
        self.execute_task_service = self.create_service(
            ExecuteTask, '/execute_task', self.execute_task_callback
        )

        # Internal state
        self.current_task_plan = None
        self.active_tasks = []
        self.task_execution_lock = threading.Lock()
        self.vla_active = False

        # Task execution thread
        self.task_execution_thread = threading.Thread(target=self.task_execution_worker, daemon=True)
        self.task_execution_thread.start()

        # Coordination timer
        coord_freq = self.get_parameter('coordination_frequency').value
        self.coordination_timer = self.create_timer(1.0/coord_freq, self.coordination_callback)

        self.node_ready = True
        self.get_logger().info('VLA Coordinator Node initialized')

    def parsed_command_callback(self, msg):
        """Handle parsed commands and coordinate VLA response"""
        self.get_logger().info(f'Received parsed command: {msg.intent} - {msg.action}')

        # Update system status
        status_msg = String()
        status_msg.data = f'Processing command: {msg.intent}'
        self.status_publisher.publish(status_msg)

    def task_plan_callback(self, msg):
        """Handle incoming task plans from cognitive planner"""
        self.get_logger().info(f'Received task plan with {len(msg.tasks)} tasks')

        with self.task_execution_lock:
            self.current_task_plan = msg
            self.active_tasks = list(msg.tasks)  # Copy tasks to active list

        # Update status
        status_msg = String()
        status_msg.data = f'New plan received: {msg.description}'
        self.status_publisher.publish(status_msg)

    def scene_callback(self, msg):
        """Handle scene understanding updates"""
        self.get_logger().debug(f'Scene update received: {len(msg.objects)} objects detected')

    def object_interaction_callback(self, msg):
        """Handle object interaction suggestions"""
        self.get_logger().debug(f'Object interactions suggested: {len(msg.interactions)} items')

    def execute_task_callback(self, request, response):
        """Execute a specific task"""
        try:
            # Validate task safety
            safety_service = self.create_client(ValidateAction, '/validate_action')
            if safety_service.wait_for_service(timeout_sec=1.0):
                safety_request = ValidateAction.Request()
                safety_request.action = request.task  # Assuming task structure matches action
                safety_future = safety_service.call_async(safety_request)

                # Wait for safety validation
                rclpy.spin_until_future_complete(self, safety_future, timeout_sec=2.0)
                safety_result = safety_future.result()

                if not safety_result.is_safe:
                    response.success = False
                    response.message = f'Safety validation failed: {safety_result.safety_issues}'
                    return response
            else:
                self.get_logger().warn('Safety validation service not available')

            # Execute the task based on action type
            execution_success = self.execute_task_by_type(request.task)

            response.success = execution_success
            response.message = 'Task executed' if execution_success else 'Task execution failed'

        except Exception as e:
            self.get_logger().error(f'Error executing task: {e}')
            response.success = False
            response.message = f'Execution error: {str(e)}'

        return response

    def execute_task_by_type(self, task) -> bool:
        """Execute task based on its action type"""
        action_type = task.action_type.lower()

        if action_type == 'navigation':
            return self.execute_navigation_task(task)
        elif action_type == 'manipulation':
            return self.execute_manipulation_task(task)
        elif action_type == 'perception':
            return self.execute_perception_task(task)
        elif action_type == 'communication':
            return self.execute_communication_task(task)
        elif action_type == 'wait':
            return self.execute_wait_task(task)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation_task(self, task) -> bool:
        """Execute navigation task"""
        try:
            if hasattr(task, 'target_pose'):
                target_pose = task.target_pose.pose

                # Simple navigation to target (in simulation)
                # In real implementation, this would use navigation2
                cmd_msg = Twist()

                # Calculate direction to target (simplified)
                # This is a basic example - real navigation would use Nav2
                cmd_msg.linear.x = 0.2  # Move forward slowly
                cmd_msg.angular.z = 0.1  # Small turn for demonstration

                # Publish command for a short duration
                for i in range(50):  # 5 seconds at 10Hz
                    self.action_command_publisher.publish(cmd_msg)
                    time.sleep(0.1)

                # Stop
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = 0.0
                self.action_command_publisher.publish(cmd_msg)

                self.get_logger().info(f'Navigation task completed to {target_pose.position.x:.2f}, {target_pose.position.y:.2f}')
                return True

        except Exception as e:
            self.get_logger().error(f'Error in navigation task: {e}')
            return False

    def execute_manipulation_task(self, task) -> bool:
        """Execute manipulation task"""
        try:
            # In simulation, this would trigger manipulation simulation
            # In real implementation, this would control robot arms/grippers
            target_object = task.target_object if hasattr(task, 'target_object') else 'unknown'

            self.get_logger().info(f'Manipulation task for object: {target_object}')

            # Simulate manipulation actions
            # This would interface with actual manipulation stack in real implementation
            time.sleep(2.0)  # Simulate manipulation time

            self.get_logger().info(f'Manipulation task completed for {target_object}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in manipulation task: {e}')
            return False

    def execute_perception_task(self, task) -> bool:
        """Execute perception task"""
        try:
            # Perception tasks typically involve waiting for sensor data
            # or triggering specific sensor behaviors
            self.get_logger().info('Perception task executed - analyzing environment')

            # In real implementation, this might trigger specific perception behaviors
            # or wait for specific sensor data

            time.sleep(1.0)  # Simulate perception time
            return True

        except Exception as e:
            self.get_logger().error(f'Error in perception task: {e}')
            return False

    def execute_communication_task(self, task) -> bool:
        """Execute communication task"""
        try:
            # In real implementation, this would interface with speech synthesis
            # or other communication systems
            if hasattr(task, 'action_parameters'):
                params = json.loads(task.action_parameters)
                message = params.get('message', 'Default message')
            else:
                message = 'Hello, I am your robot assistant'

            self.get_logger().info(f'Communication task: {message}')

            # Publish system response
            response_msg = String()
            response_msg.data = message
            self.system_response_publisher.publish(response_msg)

            return True

        except Exception as e:
            self.get_logger().error(f'Error in communication task: {e}')
            return False

    def execute_wait_task(self, task) -> bool:
        """Execute wait task"""
        try:
            duration = task.estimated_duration if hasattr(task, 'estimated_duration') else 1.0
            self.get_logger().info(f'Waiting task for {duration:.2f} seconds')
            time.sleep(duration)
            return True

        except Exception as e:
            self.get_logger().error(f'Error in wait task: {e}')
            return False

    def task_execution_worker(self):
        """Worker thread for executing task plans"""
        while rclpy.ok():
            try:
                with self.task_execution_lock:
                    if self.active_tasks and self.vla_active:
                        # Execute next task in plan
                        current_task = self.active_tasks.pop(0)

                        # Execute task
                        success = self.execute_task_by_type(current_task)

                        if success:
                            self.get_logger().info(f'Task completed: {current_task.description}')
                        else:
                            self.get_logger().error(f'Task failed: {current_task.description}')
                            # Handle failure - maybe add to retry queue or abort plan
                            break

                time.sleep(0.1)  # Small delay between task checks

            except Exception as e:
                self.get_logger().error(f'Error in task execution worker: {e}')
                time.sleep(1.0)

    def coordination_callback(self):
        """Periodic coordination callback"""
        # Monitor system status and coordinate activities
        status_msg = String()

        with self.task_execution_lock:
            if self.active_tasks:
                status_msg.data = f'Executing plan: {len(self.active_tasks)} tasks remaining'
            elif self.current_task_plan:
                status_msg.data = f'Plan completed: {self.current_task_plan.description}'
            else:
                status_msg.data = 'Waiting for tasks'

        self.status_publisher.publish(status_msg)

    def start_vla_system(self):
        """Start the VLA system"""
        self.vla_active = True
        self.get_logger().info('VLA system activated')

    def stop_vla_system(self):
        """Stop the VLA system"""
        self.vla_active = False
        self.get_logger().info('VLA system deactivated')
```

### 2.2 Isaac ROS Integration

Integrate Isaac ROS components for enhanced perception:

```python
# isaac_ros_integration.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from capstone_system_interfaces.msg import EnhancedPerceptionData
from capstone_system_interfaces.srv import ProcessIsaacData
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import Dict, Any, Optional

class IsaacROSIntegrationNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('enable_isaac_processing', True)
        self.declare_parameter('gpu_device_id', 0)
        self.declare_parameter('tensorrt_engine_path', '')

        # Publishers
        self.enhanced_perception_publisher = self.create_publisher(
            EnhancedPerceptionData, '/isaac_ros/enhanced_perception', 10
        )

        # Subscribers - these would connect to Isaac ROS output topics
        self.rgb_image_subscriber = self.create_subscription(
            Image, '/isaac_ros/rgb_image', self.rgb_image_callback, 10
        )
        self.depth_image_subscriber = self.create_subscription(
            Image, '/isaac_ros/depth_image', self.depth_image_callback, 10
        )
        self.detections_subscriber = self.create_subscription(
            Detection2DArray, '/isaac_ros/detections', self.detections_callback, 10
        )

        # Services
        self.process_isaac_data_service = self.create_service(
            ProcessIsaacData, '/process_isaac_data', self.process_isaac_data_callback
        )

        # Isaac ROS pipeline components (these would be actual Isaac ROS nodes in real implementation)
        self.isaac_initialized = False
        self.depth_data = None
        self.rgb_data = None
        self.detection_results = None

        # Performance monitoring
        self.processing_times = []
        self.average_processing_time = 0.0

        # Initialize Isaac ROS components (mock implementation for this example)
        self.initialize_isaac_components()

        self.node_ready = True
        self.get_logger().info('Isaac ROS Integration Node initialized')

    def initialize_isaac_components(self):
        """Initialize Isaac ROS pipeline components"""
        # In a real implementation, this would initialize actual Isaac ROS nodes
        # For this example, we'll simulate the functionality
        self.isaac_initialized = True
        self.get_logger().info('Isaac ROS components initialized (simulated)')

    def rgb_image_callback(self, msg):
        """Process RGB image from Isaac ROS pipeline"""
        if not self.isaac_initialized:
            return

        try:
            # Store RGB data for enhanced processing
            self.rgb_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # If we have depth data, process them together
            if self.depth_data is not None:
                self.process_multimodal_data(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_image_callback(self, msg):
        """Process depth image from Isaac ROS pipeline"""
        if not self.isaac_initialized:
            return

        try:
            # Store depth data for enhanced processing
            self.depth_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # If we have RGB data, process them together
            if self.rgb_data is not None:
                self.process_multimodal_data(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def detections_callback(self, msg):
        """Process detections from Isaac ROS pipeline"""
        if not self.isaac_initialized:
            return

        try:
            # Store detection results
            self.detection_results = msg

            # Process with current RGB and depth data if available
            if self.rgb_data is not None and self.depth_data is not None:
                self.process_multimodal_data_with_detections(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing detections: {e}')

    def process_multimodal_data(self, header):
        """Process combined RGB and depth data"""
        start_time = time.time()

        try:
            # Perform enhanced perception using Isaac ROS techniques
            enhanced_data = self.perform_enhanced_perception(
                self.rgb_data, self.depth_data
            )

            # Create and publish enhanced perception message
            enhanced_msg = self.create_enhanced_perception_message(enhanced_data, header)
            self.enhanced_perception_publisher.publish(enhanced_msg)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            self.average_processing_time = sum(self.processing_times) / len(self.processing_times)

            self.get_logger().debug(f'Enhanced perception processed in {processing_time:.3f}s (avg: {self.average_processing_time:.3f}s)')

        except Exception as e:
            self.get_logger().error(f'Error in multimodal processing: {e}')

    def process_multimodal_data_with_detections(self, header):
        """Process combined RGB, depth, and detection data"""
        try:
            # Combine Isaac ROS detections with our enhanced processing
            enhanced_data = self.perform_enhanced_perception_with_detections(
                self.rgb_data, self.depth_data, self.detection_results
            )

            # Create and publish enhanced perception message
            enhanced_msg = self.create_enhanced_perception_message(enhanced_data, header)
            self.enhanced_perception_publisher.publish(enhanced_msg)

        except Exception as e:
            self.get_logger().error(f'Error in multimodal processing with detections: {e}')

    def perform_enhanced_perception(self, rgb_image, depth_image) -> Dict[str, Any]:
        """Perform enhanced perception using Isaac-inspired techniques"""
        results = {
            'object_instances': [],
            'surface_normals': None,
            'semantic_segmentation': None,
            '3d_reconstruction': None,
            'spatial_affordances': [],
            'material_properties': []
        }

        # In a real Isaac ROS implementation, this would use actual Isaac ROS perception nodes
        # For this example, we'll simulate enhanced perception capabilities

        height, width = rgb_image.shape[:2]

        # Simulate object instance segmentation
        # In real implementation, this would use Isaac ROS segmentation nodes
        for i in range(np.random.randint(1, 4)):  # 1-3 objects
            obj = {
                'class': np.random.choice(['person', 'chair', 'table', 'cup']),
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox_2d': {
                    'x': int(np.random.uniform(0, width * 0.8)),
                    'y': int(np.random.uniform(0, height * 0.8)),
                    'width': int(np.random.uniform(50, width * 0.3)),
                    'height': int(np.random.uniform(50, height * 0.3))
                },
                'bbox_3d': self.estimate_3d_bbox(
                    [rgb_image.shape[1]//2, rgb_image.shape[0]//2],  # center pixel
                    depth_image[rgb_image.shape[0]//2, rgb_image.shape[1]//2] if depth_image is not None else 1.0
                ),
                'center_3d': self.pixel_to_3d(
                    [rgb_image.shape[1]//2, rgb_image.shape[0]//2],
                    depth_image[rgb_image.shape[0]//2, rgb_image.shape[1]//2] if depth_image is not None else 1.0
                )
            }
            results['object_instances'].append(obj)

        # Simulate surface normal estimation
        # In real implementation, this would use Isaac ROS normal estimation
        results['surface_normals'] = np.random.rand(height, width, 3).astype(np.float32)

        # Simulate affordance estimation
        # In real implementation, this would use Isaac ROS affordance analysis
        for obj in results['object_instances']:
            if obj['class'] in ['chair', 'table']:
                affordance = {
                    'object_class': obj['class'],
                    'affordance_type': 'sit_on' if obj['class'] == 'chair' else 'place_on',
                    'position_3d': obj['center_3d'],
                    'confidence': obj['confidence'] * 0.9
                }
                results['spatial_affordances'].append(affordance)

        return results

    def estimate_3d_bbox(self, pixel_coords, depth_value) -> Dict[str, float]:
        """Estimate 3D bounding box from 2D pixel and depth"""
        # This is a simplified estimation - real implementation would use camera intrinsics
        x_3d = (pixel_coords[0] - 320) * depth_value * 0.001  # Approximate conversion
        y_3d = (pixel_coords[1] - 240) * depth_value * 0.001
        z_3d = depth_value

        # Estimate size based on object class (simplified)
        size_x = 0.3  # meters
        size_y = 0.3
        size_z = 0.3

        return {
            'center_x': x_3d,
            'center_y': y_3d,
            'center_z': z_3d,
            'size_x': size_x,
            'size_y': size_y,
            'size_z': size_z
        }

    def pixel_to_3d(self, pixel_coords, depth_value) -> List[float]:
        """Convert pixel coordinates to 3D world coordinates"""
        # Simplified conversion - real implementation would use camera intrinsics
        x_3d = (pixel_coords[0] - 320) * depth_value * 0.001
        y_3d = (pixel_coords[1] - 240) * depth_value * 0.001
        z_3d = depth_value

        return [x_3d, y_3d, z_3d]

    def perform_enhanced_perception_with_detections(self, rgb_image, depth_image, detections) -> Dict[str, Any]:
        """Perform enhanced perception using Isaac ROS detections"""
        # Combine Isaac ROS detections with our enhanced processing
        base_results = self.perform_enhanced_perception(rgb_image, depth_image)

        # Enhance with Isaac ROS detection results
        for detection in detections.detections:
            # Add Isaac ROS detection confidence and class information
            # This would refine our perception results
            pass

        return base_results

    def create_enhanced_perception_message(self, enhanced_data, header) -> EnhancedPerceptionData:
        """Create EnhancedPerceptionData message from enhanced perception results"""
        msg = EnhancedPerceptionData()
        msg.header = header
        msg.timestamp = self.get_clock().now().to_msg()

        # Populate object instances
        for obj in enhanced_data['object_instances']:
            # Convert to message format
            pass

        # Populate affordances
        for affordance in enhanced_data['spatial_affordances']:
            # Convert to message format
            pass

        # Add performance metrics
        msg.processing_time = self.average_processing_time
        msg.confidence_score = 0.85  # Average confidence

        return msg

    def process_isaac_data_callback(self, request, response):
        """Service callback for processing Isaac ROS data"""
        try:
            # Process the requested Isaac ROS data
            if request.data_type == 'enhanced_perception':
                if self.rgb_data is not None and self.depth_data is not None:
                    enhanced_results = self.perform_enhanced_perception(
                        self.rgb_data, self.depth_data
                    )
                    response.results = json.dumps(enhanced_results)
                    response.success = True
                    response.message = 'Enhanced perception completed'
                else:
                    response.success = False
                    response.message = 'Insufficient data for processing'
            else:
                response.success = False
                response.message = f'Unknown data type: {request.data_type}'

        except Exception as e:
            self.get_logger().error(f'Error in Isaac data processing service: {e}')
            response.success = False
            response.message = f'Processing error: {str(e)}'

        return response
```

## 3. Hands-on Exercises

### Exercise 1: AI Perception System Implementation
**Objective:** Implement the complete AI perception system with Isaac ROS integration.

**Prerequisites:**
- Understanding of computer vision and perception concepts
- Experience with ROS 2 development
- Completion of Phase 1 and 2

**Steps:**
1. Implement the AI Perception Node with object detection and scene understanding
2. Integrate camera and sensor data processing
3. Implement 3D object localization and tracking
4. Add safety consideration analysis
5. Test perception accuracy in simulation
6. Validate real-time performance requirements
7. Document perception system architecture and performance

**Expected Outcome:** Working AI perception system that can detect objects, understand scenes, and provide 3D localization with safety analysis.

**Troubleshooting Tips:**
- Ensure proper camera calibration and intrinsics
- Check GPU availability for accelerated perception
- Validate sensor data synchronization
- Monitor real-time performance requirements

### Exercise 2: Cognitive Planning System
**Objective:** Implement the LLM-based cognitive planning system.

**Prerequisites:**
- Completed Exercise 1
- Understanding of LLM integration
- Experience with service and message handling

**Steps:**
1. Implement the Cognitive Planning Node with LLM integration
2. Create task planning and decomposition system
3. Implement safety validation for generated plans
4. Add context history management
5. Test with various command types and scenarios
6. Validate plan feasibility and safety
7. Optimize for real-time performance

**Expected Outcome:** Cognitive planning system that generates safe, feasible task plans from natural language commands.

### Exercise 3: VLA Coordination System
**Objective:** Create the Vision-Language-Action coordination system.

**Prerequisites:**
- Completed previous exercises
- Understanding of system integration
- Experience with complex ROS 2 systems

**Steps:**
1. Implement the VLA Coordinator Node
2. Create task execution and management system
3. Implement action type handlers (navigation, manipulation, etc.)
4. Add safety validation and monitoring
5. Test end-to-end VLA functionality
6. Validate system performance and reliability
7. Document coordination protocols and safety measures

**Expected Outcome:** Complete VLA system that coordinates vision, language, and action capabilities for autonomous behavior.

## 4. Safety and Ethical Considerations

When implementing AI-Robot Brain and VLA systems:
- Ensure all AI-generated plans are validated for safety before execution
- Implement multiple safety layers and validation checks
- Consider bias in LLM outputs and its impact on robot behavior
- Plan for graceful degradation when AI systems fail
- Maintain human oversight for critical decisions
- Implement privacy protection for visual and linguistic data
- Ensure ethical considerations are integrated into AI decision-making
- Design systems that can explain their decisions and actions

## 5. Phase Summary

In this phase, you've completed:
- Implementation of AI perception systems with Isaac ROS integration
- Development of LLM-based cognitive planning and reasoning
- Creation of Vision-Language-Action coordination system
- Integration of multimodal AI capabilities for robotic applications
- Implementation of safety validation for AI-generated plans
- Validation of AI system performance and real-time requirements

The AI-Robot Brain and VLA integration provides the cognitive capabilities that enable your humanoid robot to understand natural language commands, perceive its environment, and execute complex tasks autonomously.

## 6. Assessment Questions

### Multiple Choice
1. What is the primary function of the VLA Coordinator Node?
   a) To process visual data only
   b) To coordinate between vision, language, and action systems
   c) To handle speech recognition
   d) To manage navigation only

   Answer: b) To coordinate between vision, language, and action systems

2. Which component validates the safety of AI-generated plans?
   a) Voice recognition system
   b) Cognitive planning system
   c) Safety validator in cognitive planner
   d) Navigation system

   Answer: c) Safety validator in cognitive planner

### Practical Questions
1. Implement a complete VLA system that accepts natural language commands, perceives the environment visually, and executes appropriate actions with safety validation and real-time performance.

## 7. Next Steps

After completing Phase 4, you should:
- Validate AI system performance in various scenarios
- Test safety mechanisms thoroughly
- Optimize system performance for real-time operation
- Prepare for Phase 5: System Integration and Validation
- Document AI system capabilities and limitations
- Plan for real-world deployment considerations

The AI-Robot Brain and VLA systems you've implemented provide the cognitive foundation for autonomous humanoid robot behavior, enabling natural interaction and intelligent task execution.