---
sidebar_position: 3
---

# Chapter 2: Topics and Services

## Introduction

Communication between nodes is the backbone of any ROS 2 system. Topics and services provide the two primary communication patterns that enable nodes to exchange data and request specific actions. Understanding these patterns is essential for building effective robotic applications.

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement publisher-subscriber communication using topics
- Create and use services for request-response communication
- Understand Quality of Service (QoS) settings and their impact
- Design effective message schemas for your applications
- Apply communication patterns to solve common robotic challenges

## Topics: Publisher-Subscriber Pattern

Topics enable asynchronous, many-to-many communication between nodes. Publishers send messages to topics, and subscribers receive messages from topics. This decoupled communication pattern is ideal for continuous data streams.

### Topic Characteristics
- **Asynchronous**: Publishers and subscribers don't need to run simultaneously
- **Many-to-many**: Multiple publishers can send to a topic, multiple subscribers can receive from it
- **Data-driven**: Communication is triggered by data availability
- **Unidirectional**: Data flows from publisher to subscriber

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

### Creating a Subscriber

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Services: Request-Response Pattern

Services provide synchronous, request-response communication between nodes. A client sends a request to a service and waits for a response. This pattern is ideal for operations that require immediate feedback.

### Service Characteristics
- **Synchronous**: Client waits for response from service
- **One-to-one**: One client communicates with one service server
- **Action-oriented**: Communication is triggered by specific requests
- **Bidirectional**: Request and response data flow

### Creating a Service Server

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Creating a Service Client

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Quality of Service (QoS) Profiles

QoS profiles allow you to configure how messages are delivered between publishers and subscribers. Different communication needs require different QoS settings.

### QoS Policies

1. **Reliability**: Ensures message delivery
   - `RELIABLE`: All messages are delivered (with retries)
   - `BEST_EFFORT`: Messages are delivered if possible

2. **Durability**: Persistence of messages for late-joining subscribers
   - `TRANSIENT_LOCAL`: Messages persist for late joiners
   - `VOLATILE`: No message persistence

3. **History**: How many messages to store
   - `KEEP_LAST`: Store the most recent messages
   - `KEEP_ALL`: Store all messages (limited by memory)

4. **Depth**: Number of messages to store when using KEEP_LAST

### QoS Examples

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Reliable communication for critical data
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Best-effort for high-frequency sensor data
best_effort_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Persistent messages for configuration
persistent_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## Practical Exercise: Sensor Data Publisher

### Task
Create a node that simulates sensor data (temperature readings) and publishes them to a topic. Create another node that subscribes to this topic and processes the data.

### Requirements
1. Publisher node: Simulate temperature readings with random noise
2. Subscriber node: Calculate and log average temperature over time
3. Use appropriate QoS settings for sensor data
4. Add parameter for sensor frequency

### Solution - Publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import random

class TemperaturePublisher(Node):

    def __init__(self):
        super().__init__('temperature_publisher')

        # Declare parameters
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('base_temperature', 25.0)

        rate = self.get_parameter('publish_rate').value
        self.base_temp = self.get_parameter('base_temperature').value

        # Create publisher with QoS for sensor data
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.publisher_ = self.create_publisher(Float64, 'temperature', sensor_qos)
        self.timer = self.create_timer(rate, self.timer_callback)

        self.get_logger().info('Temperature publisher started')

    def timer_callback(self):
        # Simulate temperature with some random variation
        temp_msg = Float64()
        temp_msg.data = self.base_temp + random.uniform(-2.0, 2.0)

        self.publisher_.publish(temp_msg)
        self.get_logger().info(f'Temperature: {temp_msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    temp_publisher = TemperaturePublisher()

    try:
        rclpy.spin(temp_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        temp_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Solution - Subscriber:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy

class TemperatureProcessor(Node):

    def __init__(self):
        super().__init__('temperature_processor')

        # Maintain running average
        self.temp_sum = 0.0
        self.temp_count = 0

        # Create subscription with matching QoS
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.subscription = self.create_subscription(
            Float64,
            'temperature',
            self.temp_callback,
            sensor_qos)

        self.get_logger().info('Temperature processor started')

    def temp_callback(self, msg):
        self.temp_sum += msg.data
        self.temp_count += 1

        avg_temp = self.temp_sum / self.temp_count if self.temp_count > 0 else 0.0

        self.get_logger().info(
            f'Received: {msg.data:.2f}°C, '
            f'Average: {avg_temp:.2f}°C, '
            f'Sample count: {self.temp_count}'
        )

def main(args=None):
    rclpy.init(args=args)
    temp_processor = TemperatureProcessor()

    try:
        rclpy.spin(temp_processor)
    except KeyboardInterrupt:
        pass
    finally:
        temp_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Communication Patterns

### Actions
Actions combine the best of topics and services for long-running tasks with feedback:

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Goal succeeded')
        return result
```

## Applications and Real-World Examples

### Mobile Robotics
- **Topics**: Sensor data streams (lidar, cameras, IMU), odometry
- **Services**: Map loading, navigation goals, calibration
- **Actions**: Navigation tasks, manipulation sequences

### Industrial Automation
- **Topics**: Conveyor belt sensors, robot joint states
- **Services**: Part inspection, quality control decisions
- **Actions**: Assembly sequences, material handling

## Safety Guidelines

1. **QoS Matching**: Ensure publisher and subscriber QoS profiles are compatible
2. **Resource Limits**: Set appropriate depth limits to prevent memory issues
3. **Error Handling**: Implement timeout mechanisms for service calls
4. **Security**: Use security policies for sensitive communication
5. **Monitoring**: Monitor communication health and performance

## Summary

In this chapter, we've covered:
- Topic-based publisher-subscriber communication
- Service-based request-response communication
- Quality of Service (QoS) profiles and their applications
- Practical examples with sensor data
- Advanced patterns like actions
- Safety guidelines for communication

These communication patterns form the foundation of ROS 2 applications. In the next chapter, we'll explore the Python client library (rclpy) in detail.