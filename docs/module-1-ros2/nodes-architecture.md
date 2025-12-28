---
sidebar_position: 2
---

# Chapter 1: Nodes and Architecture

## Introduction

Nodes are the fundamental building blocks of any ROS 2 system. A node is a process that performs computation and communicates with other nodes through messages. Understanding node architecture is crucial for developing effective robotic applications.

## Learning Objectives

By the end of this chapter, you will be able to:
- Define and create ROS 2 nodes
- Understand the node lifecycle and management
- Implement node parameters and configuration
- Use node composition techniques
- Apply best practices for node design

## What is a Node?

A node in ROS 2 is a process that performs computation. Nodes are the fundamental building blocks of ROS 2 applications. Each node is typically responsible for a specific task or set of related tasks within the larger robotic system.

### Node Characteristics
- **Modularity**: Each node performs a specific function
- **Communication**: Nodes communicate through topics, services, and actions
- **Independence**: Nodes can run in separate processes or threads
- **Lifecycle**: Nodes have a defined lifecycle managed by ROS 2

## Creating a Node

Let's explore how to create a basic ROS 2 node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
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

## Node Architecture Components

### 1. Node Container
The node container manages the execution of nodes and their lifecycle. It handles:
- Node initialization and destruction
- Parameter management
- Logging configuration
- Clock management

### 2. Executors
Executors manage the execution of callbacks for nodes:
- **Single-threaded Executor**: Executes callbacks sequentially
- **Multi-threaded Executor**: Executes callbacks in parallel
- **Static Single-threaded Executor**: Optimized for static node sets

### 3. Node Parameters
Nodes can be configured through parameters:
- Runtime configuration
- Type safety
- Parameter validation
- Parameter callbacks

## Practical Exercise: Creating Your First Node

### Task
Create a ROS 2 node that publishes the current time every second.

### Requirements
1. Create a node called "time_publisher"
2. Publish the current time to a topic called "current_time"
3. Include a parameter for the publishing rate (default: 1 second)
4. Add logging to track published messages

### Solution
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import time

class TimePublisher(Node):

    def __init__(self):
        super().__init__('time_publisher')

        # Declare parameter with default value
        self.declare_parameter('publish_rate', 1.0)
        rate = self.get_parameter('publish_rate').value

        # Create publisher
        self.publisher_ = self.create_publisher(String, 'current_time', 10)

        # Create timer based on parameter
        self.timer = self.create_timer(rate, self.timer_callback)

        self.get_logger().info(f'Time publisher started with rate: {rate}s')

    def timer_callback(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        msg = String()
        msg.data = f'Current time: {current_time}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    time_publisher = TimePublisher()

    try:
        rclpy.spin(time_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        time_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Node Concepts

### Node Composition
Node composition allows multiple nodes to run within the same process, reducing communication overhead:

```python
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class ComposedNode(Node):
    def __init__(self):
        super().__init__('composed_node')

        # Create multiple publishers/subscribers within one node
        self.pub1 = self.create_publisher(String, 'topic1', 10)
        self.pub2 = self.create_publisher(Int64, 'topic2', 10)

        # Use callback groups for parallel execution
        cb_group = MutuallyExclusiveCallbackGroup()
        self.sub1 = self.create_subscription(String, 'input1',
                                           self.callback1, 10,
                                           callback_group=cb_group)
```

### Lifecycle Nodes
Lifecycle nodes provide explicit state management:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecyclePublisher(LifecycleNode):

    def __init__(self):
        super().__init__('lifecycle_publisher')

    def on_configure(self, state: LifecycleState):
        self.pub = self.create_publisher(String, 'topic', 10)
        return TransitionCallbackReturn.SUCCESS
```

## Applications and Real-World Examples

### Industrial Robotics
In manufacturing environments, nodes handle:
- Sensor data processing
- Motion control
- Safety monitoring
- Quality assurance

### Autonomous Vehicles
Nodes in autonomous vehicles manage:
- Perception systems (lidar, cameras, radar)
- Planning and navigation
- Control systems
- Communication with infrastructure

## Safety Guidelines

1. **Resource Management**: Always properly clean up resources in node destruction
2. **Error Handling**: Implement robust error handling in callbacks
3. **Logging**: Use appropriate logging levels for debugging and monitoring
4. **Parameter Validation**: Validate parameters before using them
5. **Threading Safety**: Ensure thread safety when using multi-threaded executors

## Summary

In this chapter, we've covered the fundamental concepts of ROS 2 nodes:
- Nodes as the basic building blocks of ROS 2 systems
- How to create and configure nodes
- Advanced concepts like node composition and lifecycle nodes
- Practical examples and exercises
- Safety guidelines for node development

In the next chapter, we'll explore topics and services for inter-node communication.