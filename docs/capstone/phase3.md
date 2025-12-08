---
sidebar_position: 4
learning_objectives:
  - Integrate Digital Twin simulation environment with the capstone system
  - Implement Gazebo simulation for robot development and testing
  - Create Unity integration for advanced visualization (if applicable)
  - Develop physics and sensor simulation for realistic testing
  - Validate system behavior in simulated environments
  - Prepare for real-world deployment through simulation
prerequisites:
  - Completion of Phase 1 (System Architecture) and Phase 2 (ROS 2 Framework)
  - Understanding of Gazebo and simulation concepts
  - Experience with robotics simulation environments
  - Access to appropriate simulation software (Gazebo, Unity if applicable)
estimated_time: "12 hours"
---

# Phase 3: Digital Twin and Simulation Environment Integration

## Learning Objectives

By completing this phase, you will be able to:
- Integrate Digital Twin simulation environment with the capstone system
- Implement Gazebo simulation for comprehensive robot development and testing
- Create Unity integration for advanced visualization and AI training (if applicable)
- Develop realistic physics and sensor simulation for comprehensive testing
- Validate system behavior in simulated environments before real-world deployment
- Establish simulation-to-reality transfer protocols and validation procedures
- Ensure the simulation environment meets all system requirements

## Introduction

Phase 3 focuses on implementing the Digital Twin simulation environment that will serve as the primary development and testing platform for your capstone system. This phase integrates the simulation capabilities from Module 2 with the communication framework developed in Phase 2, creating a comprehensive virtual environment where you can develop, test, and validate your autonomous humanoid robot system before real-world deployment.

The Digital Twin environment is crucial because it:
- Provides a safe environment for system development and testing
- Enables rapid iteration and debugging without physical hardware risks
- Allows for controlled testing of safety mechanisms and edge cases
- Facilitates AI model development and validation
- Supports the simulation-to-reality transfer of learned behaviors
- Validates system performance before physical deployment

## 1. Gazebo Simulation Integration

### 1.1 Robot Model Definition

Create a comprehensive robot model for simulation:

```xml
<!-- capstone_robot.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="capstone_humanoid">

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.08" />
  <xacro:property name="base_mass" value="50.0" />
  <xacro:property name="wheel_mass" value="5.0" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.8 0.6 1.0"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.8 0.6 1.0"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <mass value="${base_mass}"/>
      <inertia
        ixx="1.0" ixy="0.0" ixz="0.0"
        iyy="1.0" iyz="0.0"
        izz="1.0"/>
    </inertial>
  </link>

  <!-- Head Link -->
  <link name="head_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia
        ixx="0.01" ixy="0.0" ixz="0.0"
        iyy="0.01" iyz="0.0"
        izz="0.01"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
  </joint>

  <!-- Camera Mount -->
  <link name="camera_mount">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head_link"/>
    <child link="camera_mount"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <!-- RGB Camera -->
  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="camera_optical_joint" type="fixed">
    <parent link="camera_mount"/>
    <child link="camera_link"/>
    <origin xyz="0 0 0" rpy="-${M_PI/2} 0 -${M_PI/2}"/>
  </joint>

  <!-- Camera Gazebo Plugin -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>100.0</max_depth>
        <robot_namespace>/capstone_robot</robot_namespace>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU Sensor -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <robotNamespace>/capstone_robot</robotNamespace>
        <topicName>imu/data</topicName>
        <serviceName>imu/service</serviceName>
        <gaussianNoise>0.01</gaussianNoise>
        <accelDrift>0.001 0.001 0.001</accelDrift>
        <accelDriftFrequency>0.001 0.001 0.001</accelDriftFrequency>
        <accelGaussianNoise>0.001 0.001 0.001</accelGaussianNoise>
        <rateDrift>0.001 0.001 0.001</rateDrift>
        <rateDriftFrequency>0.001 0.001 0.001</rateDriftFrequency>
        <rateGaussianNoise>0.001 0.001 0.001</rateGaussianNoise>
        <headingDrift>0.001 0.001 0.001</headingDrift>
        <headingDriftFrequency>0.001 0.001 0.001</headingDriftFrequency>
        <headingGaussianNoise>0.001 0.001 0.001</headingGaussianNoise>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Hokuyo Laser Scanner -->
  <link name="laser_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0.3 0 0.2" rpy="0 0 0"/>
  </joint>

  <gazebo reference="laser_link">
    <sensor type="ray" name="head_hokuyo_sensor">
      <pose>0 0 0 0 0 0</pose>
      <visualize>false</visualize>
      <update_rate>40</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </ray>
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <topicName>/capstone_robot/scan</topicName>
        <frameName>laser_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Differential Drive Plugin -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <commandTopic>/capstone_robot/cmd_vel</commandTopic>
      <odometryTopic>/capstone_robot/odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <odometrySource>world</odometrySource>
      <publishOdomTF>true</publishOdomTF>
      <robotBaseFrame>base_link</robotBaseFrame>
      <publishTf>true</publishTf>
      <publishWheelTF>false</publishWheelTF>
      <publishWheelJointState>false</publishWheelJointState>
      <legacyMode>false</legacyMode>
      <updateRate>30</updateRate>
      <leftJoint>left_wheel_joint</leftJoint>
      <rightJoint>right_wheel_joint</rightJoint>
      <wheelSeparation>0.6</wheelSeparation>
      <wheelDiameter>0.2</wheelDiameter>
      <broadcastTF>1</broadcastTF>
      <wheelTorque>20</wheelTorque>
      <wheelAcceleration>1.8</wheelAcceleration>
      <rosDebugLevel>na</rosDebugLevel>
    </plugin>
  </gazebo>

</robot>
```

### 1.2 Gazebo World Definition

Create a comprehensive simulation world:

```xml
<!-- capstone_world.world -->
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="capstone_world">

    <!-- Include the outdoor playground world -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add walls to create a room -->
    <model name='wall_1'>
      <pose>0 -5 1 0 0 0</pose>
      <link name='link'>
        <pose>0 0 1 0 0 0</pose>
        <collision name='collision'>
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <kinematic>0</kinematic>
        <gravity>1</gravity>
        <enable_wind>0</enable_wind>
        <velocity_decay>
          <linear>0</linear>
          <angular>0.01</angular>
        </velocity_decay>
        <mass>1</mass>
        <inertia>
          <ixx>1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>1</iyy>
          <iyz>0</iyz>
          <izz>1</izz>
        </inertia>
      </link>
    </model>

    <model name='wall_2'>
      <pose>0 5 1 0 0 0</pose>
      <link name='link'>
        <pose>0 0 1 0 0 0</pose>
        <collision name='collision'>
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>20 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name='wall_3'>
      <pose>-10 0 1 0 0 1.5707</pose>
      <link name='link'>
        <pose>0 0 1 0 0 0</pose>
        <collision name='collision'>
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name='wall_4'>
      <pose>10 0 1 0 0 1.5707</pose>
      <link name='link'>
        <pose>0 0 1 0 0 0</pose>
        <collision name='collision'>
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.4 0.4 1</ambient>
            <diffuse>0.4 0.4 0.4 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add furniture and objects for realistic environment -->
    <model name='table_1'>
      <pose>-3 2 0.4 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <model name='chair_1'>
      <pose>-3 3 0.2 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Place the robot in the world -->
    <include>
      <uri>model://capstone_humanoid</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>

    <!-- Lighting -->
    <light name='spotlight' type='spot'>
      <pose>5 5 8 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.1</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
      <direction>-0.5 -0.5 -1</direction>
      <spot>
        <inner_angle>0.1</inner_angle>
        <outer_angle>0.5</outer_angle>
        <falloff>1</falloff>
      </spot>
    </light>

  </world>
</sdf>
```

### 1.3 Gazebo Launch Files

Create launch files to start the simulation environment:

```python
# launch/capstone_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    gui = LaunchConfiguration('gui')

    # Declare launch arguments
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([
            FindPackageShare('capstone_system'),
            'worlds',
            'capstone_world.world'
        ]),
        description='Choose one of the world files from `/capstone_system/worlds`'
    )

    declare_headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='False',
        description='Whether to execute gzclient)'
    )

    declare_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Whether to launch the GUI'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world,
            'gui': gui,
            'headless': headless,
            'verbose': 'false',
        }.items()
    )

    # Robot spawn node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'capstone_humanoid',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
        ],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(PathJoinSubstitution([
                FindPackageShare('capstone_system'),
                'urdf',
                'capstone_robot.urdf.xacro'
            ]).perform({})).read()
        }]
    )

    # Joint state publisher (for non-simulated joints)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_arg)
    ld.add_action(declare_world_arg)
    ld.add_action(declare_headless_arg)
    ld.add_action(declare_gui_arg)

    # Add nodes and launch descriptions
    ld.add_action(gazebo)
    ld.add_action(spawn_entity)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)

    return ld
```

## 2. Simulation Integration with ROS 2 Framework

### 2.1 Simulation Interface Node

Create a node that bridges the simulation environment with the ROS 2 framework:

```python
# simulation_interface_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from capstone_system_interfaces.srv import ValidateAction
import threading
import time
from typing import Dict, Any, Optional

class SimulationInterfaceNode(Node):
    def __init__(self):
        super().__init__('simulation_interface_node')

        # Declare parameters
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('simulation_speed', 1.0)
        self.declare_parameter('gravity_enabled', True)
        self.declare_parameter('real_time_factor', 1.0)

        # Publishers for simulation state
        self.simulation_state_publisher = self.create_publisher(
            String, '/simulation/state', 10
        )
        self.simulation_clock_publisher = self.create_publisher(
            Time, '/clock', 10
        )

        # Subscribers for robot commands
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/capstone_robot/cmd_vel', self.cmd_vel_callback, 10
        )

        # Service clients for Gazebo services
        self.reset_simulation_client = self.create_client(
            Empty, '/reset_simulation'
        )
        self.pause_simulation_client = self.create_client(
            Empty, '/pause_physics'
        )
        self.unpause_simulation_client = self.create_client(
            Empty, '/unpause_physics'
        )

        # Internal state
        self.simulation_active = True
        self.simulation_paused = False
        self.last_command_time = self.get_clock().now()
        self.robot_pose = PoseStamped()

        # Setup simulation monitoring
        self.simulation_timer = self.create_timer(0.1, self.simulation_monitor_callback)

        # Setup clock publisher if using sim time
        if self.get_parameter('use_sim_time').value:
            self.clock_timer = self.create_timer(0.001, self.publish_simulation_clock)  # 1kHz clock

        self.get_logger().info('Simulation Interface Node initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from the system"""
        # In simulation, these commands are handled by Gazebo's diff drive plugin
        # but we can monitor and validate them here
        self.last_command_time = self.get_clock().now()

        # Validate command for safety (even in simulation)
        if self.is_command_safe(msg):
            self.get_logger().debug(f'Received safe command: linear={msg.linear.x}, angular={msg.angular.z}')
        else:
            self.get_logger().warn(f'Potentially unsafe command: linear={msg.linear.x}, angular={msg.angular.z}')

    def is_command_safe(self, cmd_vel: Twist) -> bool:
        """Validate velocity command for safety"""
        # Define safe limits
        max_linear = 1.0  # m/s
        max_angular = 1.5  # rad/s

        return (abs(cmd_vel.linear.x) <= max_linear and
                abs(cmd_vel.angular.z) <= max_angular)

    def simulation_monitor_callback(self):
        """Monitor simulation state and performance"""
        current_time = self.get_clock().now()

        # Check if simulation is responsive
        time_since_last_cmd = (current_time - self.last_command_time).nanoseconds / 1e9

        if time_since_last_cmd > 5.0:  # No commands in 5 seconds
            self.get_logger().info('Simulation running, no recent commands')

    def publish_simulation_clock(self):
        """Publish simulation clock message"""
        if self.get_parameter('use_sim_time').value:
            current_time_msg = self.get_clock().now().to_msg()
            self.simulation_clock_publisher.publish(current_time_msg)

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        if self.reset_simulation_client.service_is_ready():
            future = self.reset_simulation_client.call_async(Empty.Request())
            self.get_logger().info('Simulation reset requested')
        else:
            self.get_logger().warn('Reset simulation service not available')

    def pause_simulation(self):
        """Pause the simulation"""
        if self.pause_simulation_client.service_is_ready():
            future = self.pause_simulation_client.call_async(Empty.Request())
            self.simulation_paused = True
            self.get_logger().info('Simulation paused')
        else:
            self.get_logger().warn('Pause simulation service not available')

    def unpause_simulation(self):
        """Unpause the simulation"""
        if self.unpause_simulation_client.service_is_ready():
            future = self.unpause_simulation_client.call_async(Empty.Request())
            self.simulation_paused = False
            self.get_logger().info('Simulation unpaused')
        else:
            self.get_logger().warn('Unpause simulation service not available')

    def get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'active': self.simulation_active,
            'paused': self.simulation_paused,
            'time': self.get_clock().now().to_msg(),
            'last_command_time': self.last_command_time.to_msg(),
            'real_time_factor': self.get_parameter('real_time_factor').value
        }
```

### 2.2 Physics and Sensor Simulation Validation

Implement validation nodes to ensure simulation accuracy:

```python
# simulation_validation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time
from collections import deque
from typing import Dict, List, Tuple

class SimulationValidationNode(Node):
    def __init__(self):
        super().__init__('simulation_validation_node')

        # Declare validation parameters
        self.declare_parameter('validation_frequency', 10.0)  # Hz
        self.declare_parameter('acceptable_deviation_threshold', 0.1)  # meters
        self.declare_parameter('sensor_noise_tolerance', 0.05)  # variance threshold
        self.declare_parameter('physics_accuracy_threshold', 0.02)  # meters

        # Publishers for validation results
        self.accuracy_report_publisher = self.create_publisher(
            String, '/simulation/accuracy_report', 10
        )
        self.deviation_publisher = self.create_publisher(
            Float32, '/simulation/deviation', 10
        )
        self.performance_publisher = self.create_publisher(
            String, '/simulation/performance_metrics', 10
        )

        # Subscribers for validation data
        self.odom_subscriber = self.create_subscription(
            Odometry, '/capstone_robot/odom', self.odom_callback, 10
        )
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/capstone_robot/scan', self.scan_callback, 10
        )
        self.imu_subscriber = self.create_subscription(
            Imu, '/capstone_robot/imu/data', self.imu_callback, 10
        )
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, '/capstone_robot/cmd_vel', self.cmd_vel_callback, 10
        )

        # Internal validation state
        self.odom_history = deque(maxlen=100)
        self.scan_history = deque(maxlen=10)
        self.imu_history = deque(maxlen=50)
        self.cmd_history = deque(maxlen=50)

        self.initial_pose = None
        self.last_validation_time = self.get_clock().now()
        self.validation_results = {
            'physics_accuracy': 0.0,
            'sensor_fidelity': 0.0,
            'timing_precision': 0.0,
            'overall_score': 0.0
        }

        # Setup validation timer
        validation_freq = self.get_parameter('validation_frequency').value
        self.validation_timer = self.create_timer(1.0/validation_freq, self.validation_callback)

        self.get_logger().info('Simulation Validation Node initialized')

    def odom_callback(self, msg):
        """Process odometry data for validation"""
        self.odom_history.append({
            'timestamp': msg.header.stamp,
            'pose': msg.pose.pose,
            'twist': msg.twist.twist
        })

        # Store initial pose if not set
        if self.initial_pose is None:
            self.initial_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan data for validation"""
        self.scan_history.append({
            'timestamp': msg.header.stamp,
            'ranges': np.array(msg.ranges),
            'intensities': np.array(msg.intensities) if len(msg.intensities) > 0 else None,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        })

    def imu_callback(self, msg):
        """Process IMU data for validation"""
        self.imu_history.append({
            'timestamp': msg.header.stamp,
            'orientation': msg.orientation,
            'angular_velocity': msg.angular_velocity,
            'linear_acceleration': msg.linear_acceleration
        })

    def cmd_vel_callback(self, msg):
        """Process command velocity for validation"""
        self.cmd_history.append({
            'timestamp': self.get_clock().now().to_msg(),
            'cmd_vel': msg
        })

    def validation_callback(self):
        """Perform comprehensive simulation validation"""
        current_time = self.get_clock().now()
        validation_interval = (current_time - self.last_validation_time).nanoseconds / 1e9

        if validation_interval < 1.0:  # Don't validate too frequently
            return

        # Perform different validation checks
        physics_score = self.validate_physics_accuracy()
        sensor_score = self.validate_sensor_fidelity()
        timing_score = self.validate_timing_precision()

        # Calculate overall score
        overall_score = (physics_score + sensor_score + timing_score) / 3.0

        # Update validation results
        self.validation_results = {
            'physics_accuracy': physics_score,
            'sensor_fidelity': sensor_score,
            'timing_precision': timing_score,
            'overall_score': overall_score,
            'timestamp': current_time.to_msg()
        }

        # Publish results
        self.publish_validation_results()

        # Log validation results
        self.get_logger().info(
            f'Simulation Validation - Physics: {physics_score:.2f}, '
            f'Sensor: {sensor_score:.2f}, Timing: {timing_score:.2f}, '
            f'Overall: {overall_score:.2f}'
        )

        # Check if validation results are concerning
        if overall_score < 0.7:  # Below 70% accuracy
            self.get_logger().warn(f'LOW SIMULATION ACCURACY: {overall_score:.2f}')

        self.last_validation_time = current_time

    def validate_physics_accuracy(self) -> float:
        """Validate physics simulation accuracy"""
        if len(self.odom_history) < 2:
            return 1.0  # Can't validate with insufficient data

        try:
            # Calculate expected vs actual displacement based on commands
            recent_odoms = list(self.odom_history)[-10:]  # Last 10 odom messages

            if len(recent_odoms) < 2:
                return 1.0

            # Calculate actual displacement
            start_pose = recent_odoms[0]['pose']
            end_pose = recent_odoms[-1]['pose']

            actual_displacement = np.sqrt(
                (end_pose.position.x - start_pose.position.x)**2 +
                (end_pose.position.y - start_pose.position.y)**2
            )

            # Calculate expected displacement based on commands (simplified)
            expected_displacement = 0.0
            for i in range(len(self.cmd_history)-1):
                cmd = self.cmd_history[i]['cmd_vel']
                dt = 0.1  # Assume 10Hz command rate
                expected_displacement += abs(cmd.linear.x) * dt

            # Calculate deviation
            if expected_displacement > 0:
                deviation = abs(actual_displacement - expected_displacement) / expected_displacement
                # Score decreases as deviation increases
                score = max(0.0, 1.0 - deviation)
            else:
                score = 1.0 if actual_displacement < 0.01 else 0.0  # Shouldn't move without commands

            # Publish deviation for monitoring
            deviation_msg = Float32()
            deviation_msg.data = deviation
            self.deviation_publisher.publish(deviation_msg)

            return min(score, 1.0)

        except Exception as e:
            self.get_logger().error(f'Error in physics validation: {e}')
            return 0.0

    def validate_sensor_fidelity(self) -> float:
        """Validate sensor simulation fidelity"""
        if len(self.scan_history) < 1:
            return 1.0

        try:
            recent_scans = list(self.scan_history)[-5:]  # Last 5 scans

            # Check for sensor noise consistency
            noise_levels = []
            for scan in recent_scans:
                valid_ranges = [r for r in scan['ranges'] if not (np.isnan(r) or np.isinf(r))]
                if len(valid_ranges) > 10:  # Need sufficient data points
                    range_variance = np.var(valid_ranges)
                    noise_levels.append(range_variance)

            if noise_levels:
                avg_noise = np.mean(noise_levels)
                # Higher variance indicates more noise, which might be realistic
                # but too much noise indicates simulation issues
                if avg_noise > 0.1:  # Adjust threshold as needed
                    sensor_score = 0.8  # High noise but still functional
                elif avg_noise < 0.001:  # Very little noise might indicate unrealistic simulation
                    sensor_score = 0.7  # Could be too idealistic
                else:
                    sensor_score = 0.9  # Good balance
            else:
                sensor_score = 1.0  # No data to validate

            return sensor_score

        except Exception as e:
            self.get_logger().error(f'Error in sensor validation: {e}')
            return 0.0

    def validate_timing_precision(self) -> float:
        """Validate simulation timing precision"""
        try:
            # Check if odometry messages are arriving at expected rate
            if len(self.odom_history) < 10:
                return 1.0

            # Calculate timing intervals
            timestamps = [odom['timestamp'] for odom in self.odom_history]
            intervals = []

            for i in range(1, len(timestamps)):
                interval = (timestamps[i].sec + timestamps[i].nanosec / 1e9) - \
                          (timestamps[i-1].sec + timestamps[i-1].nanosec / 1e9)
                intervals.append(interval)

            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)

                # Ideal frequency might be 50Hz (0.02s interval)
                target_interval = 0.02
                timing_error = abs(avg_interval - target_interval) / target_interval

                # Score based on timing consistency
                consistency_score = max(0.0, 1.0 - std_interval * 50)  # Adjust multiplier as needed
                accuracy_score = max(0.0, 1.0 - timing_error)

                timing_score = (consistency_score + accuracy_score) / 2.0
            else:
                timing_score = 1.0

            return timing_score

        except Exception as e:
            self.get_logger().error(f'Error in timing validation: {e}')
            return 0.0

    def publish_validation_results(self):
        """Publish comprehensive validation results"""
        # Publish accuracy report
        report_msg = String()
        report_msg.data = f"""
Simulation Validation Report:
- Physics Accuracy: {self.validation_results['physics_accuracy']:.2f}
- Sensor Fidelity: {self.validation_results['sensor_fidelity']:.2f}
- Timing Precision: {self.validation_results['timing_precision']:.2f}
- Overall Score: {self.validation_results['overall_score']:.2f}
- Timestamp: {self.validation_results['timestamp']}
        """
        self.accuracy_report_publisher.publish(report_msg)

        # Publish performance metrics
        perf_msg = String()
        perf_msg.data = f"""
Performance Metrics:
- Odom History Length: {len(self.odom_history)}
- Scan History Length: {len(self.scan_history)}
- IMU History Length: {len(self.imu_history)}
- Command History Length: {len(self.cmd_history)}
        """
        self.performance_publisher.publish(perf_msg)
```

## 3. Unity Integration (Optional Advanced Component)

### 3.1 Unity ROS Bridge Setup

If using Unity for advanced visualization, set up the bridge:

```csharp
// UnityROSBridge.cs (C# script for Unity)
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Navigation;
using RosMessageTypes.Geometry;

public class UnityROSBridge : MonoBehaviour
{
    ROSConnection ros;
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    // Robot components
    public GameObject robotBody;
    public GameObject cameraMount;
    public GameObject lidarMount;

    // Publishers and subscribers
    private string cmdVelTopic = "/capstone_robot/cmd_vel";
    private string odomTopic = "/capstone_robot/odom";
    private string imageTopic = "/capstone_robot/camera/image_raw";
    private string scanTopic = "/capstone_robot/scan";

    // Robot state
    private Vector3 robotPosition = Vector3.zero;
    private Quaternion robotRotation = Quaternion.identity;
    private float linearVelocity = 0f;
    private float angularVelocity = 0f;

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIPAddress, rosPort);

        // Subscribe to robot commands
        ros.Subscribe<TwistMsg>(cmdVelTopic, CmdVelCallback);

        // Setup publishers
        InvokeRepeating("PublishRobotState", 0.02f, 0.02f); // 50Hz
        InvokeRepeating("PublishCameraImage", 0.033f, 0.033f); // ~30Hz
    }

    void CmdVelCallback(TwistMsg msg)
    {
        linearVelocity = (float)msg.linear.x;
        angularVelocity = (float)msg.angular.z;
    }

    void Update()
    {
        // Update robot position based on velocities
        if (robotBody != null)
        {
            robotBody.transform.Translate(Vector3.forward * linearVelocity * Time.deltaTime);
            robotBody.transform.Rotate(Vector3.up, angularVelocity * Time.deltaTime * Mathf.Rad2Deg);
        }
    }

    void PublishRobotState()
    {
        if (robotBody != null)
        {
            // Create Odometry message
            OdometryMsg odomMsg = new OdometryMsg();
            odomMsg.header.stamp = new builtin_interfaces.TimeMsg();
            odomMsg.header.frame_id = "odom";

            // Set pose
            odomMsg.pose.pose.position = new geometry_msgs.PointMsg(
                robotBody.transform.position.x,
                robotBody.transform.position.y,
                robotBody.transform.position.z
            );

            Quaternion rot = robotBody.transform.rotation;
            odomMsg.pose.pose.orientation = new geometry_msgs.QuaternionMsg(
                rot.x, rot.y, rot.z, rot.w
            );

            // Set twist
            odomMsg.twist.twist.linear = new geometry_msgs.Vector3Msg(linearVelocity, 0, 0);
            odomMsg.twist.twist.angular = new geometry_msgs.Vector3Msg(0, 0, angularVelocity);

            ros.Publish(odomTopic, odomMsg);
        }
    }

    void PublishCameraImage()
    {
        if (cameraMount != null)
        {
            // Capture image from camera
            Camera cam = cameraMount.GetComponent<Camera>();
            if (cam != null)
            {
                // Render texture capture would go here
                // For now, we'll just send a placeholder
                ImageMsg imgMsg = new ImageMsg();
                imgMsg.header.stamp = new builtin_interfaces.TimeMsg();
                imgMsg.header.frame_id = "camera_link";
                imgMsg.height = 480;
                imgMsg.width = 640;
                imgMsg.encoding = "rgb8";
                imgMsg.is_bigendian = 0;
                imgMsg.step = 640 * 3; // width * bytes per pixel
                imgMsg.data = new byte[640 * 480 * 3]; // Placeholder

                ros.Publish(imageTopic, imgMsg);
            }
        }
    }

    void OnApplicationQuit()
    {
        ros.Dispose();
    }
}
```

## 4. Hands-on Exercises

### Exercise 1: Gazebo Simulation Environment Setup
**Objective:** Create and configure a comprehensive Gazebo simulation environment for the capstone robot.

**Prerequisites:**
- Understanding of URDF/Xacro format
- Experience with Gazebo simulation
- Completion of Phase 1 and 2

**Steps:**
1. Create the robot URDF model with all necessary sensors (camera, IMU, laser scanner)
2. Configure Gazebo plugins for differential drive and sensor simulation
3. Create a realistic simulation world with obstacles and features
4. Test the simulation with basic movement commands
5. Validate sensor data accuracy and reliability
6. Document the simulation setup and configuration

**Expected Outcome:** Fully functional Gazebo simulation environment with the capstone robot model and realistic sensors.

**Troubleshooting Tips:**
- Check URDF syntax and joint limits
- Verify Gazebo plugin configurations
- Ensure proper TF tree setup
- Validate sensor noise parameters

### Exercise 2: Simulation Interface Implementation
**Objective:** Implement the simulation interface that connects the simulation with the ROS 2 framework.

**Prerequisites:**
- Completed Exercise 1
- Understanding of ROS 2 simulation concepts
- Experience with Gazebo ROS integration

**Steps:**
1. Implement the SimulationInterfaceNode with proper simulation control
2. Create launch files for simulation startup
3. Test simulation control (pause, reset, speed control)
4. Validate communication between simulation and real system components
5. Test safety monitoring in simulation environment
6. Document the simulation interface functionality

**Expected Outcome:** Working simulation interface that provides proper control and monitoring of the simulation environment.

### Exercise 3: Simulation Validation System
**Objective:** Create a comprehensive validation system to ensure simulation accuracy.

**Prerequisites:**
- Completed previous exercises
- Understanding of validation concepts
- Experience with sensor and physics validation

**Steps:**
1. Implement the SimulationValidationNode with physics accuracy checks
2. Create sensor fidelity validation for realistic simulation
3. Implement timing precision validation
4. Test validation system with various scenarios
5. Validate simulation-to-reality transfer readiness
6. Document validation results and accuracy metrics

**Expected Outcome:** Comprehensive validation system that ensures simulation accuracy and readiness for real-world deployment.

## 5. Safety and Ethical Considerations

When implementing the Digital Twin and simulation environment:
- Ensure simulation safety mechanisms mirror real-world safety systems
- Validate simulation accuracy to prevent false confidence in real deployment
- Consider the ethical implications of AI training in simulation environments
- Plan for proper simulation-to-reality transfer validation
- Maintain accurate representation of safety constraints
- Implement proper monitoring and validation of simulation behavior
- Consider privacy implications of simulation data collection

## 6. Phase Summary

In this phase, you've completed:
- Implementation of comprehensive Gazebo simulation environment with realistic robot model
- Integration of sensors (camera, IMU, laser scanner) in the simulation
- Creation of realistic world environments for testing
- Development of simulation interface connecting to ROS 2 framework
- Implementation of validation systems to ensure simulation accuracy
- Preparation for simulation-to-reality transfer validation

The Digital Twin environment you've created provides a safe, controlled environment for developing and testing your capstone system before real-world deployment.

## 7. Assessment Questions

### Multiple Choice
1. What is the primary purpose of the simulation validation system?
   a) To create new simulation environments
   b) To ensure simulation accuracy and readiness for real-world deployment
   c) To control robot movement in simulation
   d) To generate sensor data

   Answer: b) To ensure simulation accuracy and readiness for real-world deployment

2. Which sensor is NOT typically simulated in Gazebo for mobile robots?
   a) Camera
   b) IMU
   c) GPS
   d) Olfactory (smell)

   Answer: d) Olfactory (smell)

### Practical Questions
1. Implement a complete Gazebo simulation environment with realistic robot model, sensors, and validation system that accurately represents the physical robot for safe development and testing.

## 8. Next Steps

After completing Phase 3, you should:
- Validate the simulation environment thoroughly with various test scenarios
- Test all system components in the simulation environment
- Verify simulation accuracy and sensor fidelity
- Prepare for Phase 4: AI-Robot Brain and VLA Integration
- Document simulation results and validation metrics
- Plan for simulation-to-reality transfer protocols

The simulation environment you've created serves as the foundation for developing and testing your complete capstone system in a safe, controlled environment before real-world deployment.