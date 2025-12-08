---
sidebar_position: 5
learning_objectives:
  - Apply ROS 2 concepts through practical exercises
  - Implement complete ROS 2 systems with multiple nodes
  - Practice debugging and troubleshooting ROS 2 applications
  - Understand best practices for ROS 2 development
prerequisites:
  - Completion of all Module 1 chapters
  - Basic Python programming knowledge
estimated_time: "3 hours"
---

# Module 1 Exercises: Robotic Nervous System (ROS 2)

## Learning Objectives

By completing these exercises, you will be able to:
- Apply ROS 2 concepts to solve practical robotics problems
- Implement complete ROS 2 systems with multiple interconnected nodes
- Practice debugging and troubleshooting techniques
- Demonstrate understanding of ROS 2 architecture and patterns

## Exercise 1: Simple Robot Control System

### Objective
Create a simple robot control system with multiple nodes that communicate using ROS 2 topics and services.

### Requirements
1. Create a **Robot Controller Node** that:
   - Publishes velocity commands to control a simulated robot
   - Subscribes to sensor feedback from the robot
   - Provides a service to set the robot's target position

2. Create a **Robot Simulator Node** that:
   - Subscribes to velocity commands
   - Publishes simulated sensor data (position, battery level, etc.)
   - Updates robot state based on velocity commands

3. Create a **Dashboard Node** that:
   - Subscribes to robot status information
   - Provides a simple text-based interface for monitoring

### Implementation Steps
1. Define custom message types for robot commands and status
2. Implement the three nodes with appropriate publishers, subscribers, and services
3. Test the system by sending commands and observing responses
4. Add parameter configuration for different robot behaviors

### Expected Outcome
A functional multi-node ROS 2 system that demonstrates the publisher-subscriber and service patterns.

### Assessment Rubric
- **Functionality** (40%): All nodes communicate correctly and system operates as expected
- **Code Quality** (30%): Proper use of ROS 2 patterns, error handling, and documentation
- **Architecture** (20%): Appropriate use of topics, services, and node organization
- **Parameter Configuration** (10%): Proper use of ROS 2 parameters for configuration

## Exercise 2: Parameter-Based Navigation System

### Objective
Implement a navigation system that uses parameters to configure different navigation behaviors.

### Requirements
1. Create a **Navigation Node** that:
   - Uses parameters to configure navigation speed, turning radius, and safety margins
   - Subscribes to goal positions
   - Publishes path planning commands
   - Implements parameter callbacks to update behavior at runtime

2. Create a **Goal Publisher Node** that:
   - Reads goal positions from a configuration file
   - Publishes goals to the navigation system
   - Handles navigation completion and moves to next goal

3. Implement parameter validation and safety checks

### Implementation Steps
1. Design parameter structure for navigation configuration
2. Implement parameter callbacks to handle runtime updates
3. Add safety validation for parameter values
4. Test parameter changes during runtime without restarting nodes

### Expected Outcome
A navigation system that can be reconfigured at runtime through parameters while maintaining safe operation.

### Assessment Rubric
- **Parameter Management** (40%): Proper implementation of parameters and callbacks
- **Runtime Configuration** (30%): System responds correctly to parameter changes during operation
- **Safety Validation** (20%): Proper validation of parameter values to ensure safe operation
- **Code Quality** (10%): Clean, well-documented implementation

## Exercise 3: Multi-Threaded Node with Actions

### Objective
Create a complex node that uses multiple threads and implements ROS 2 actions for long-running tasks.

### Requirements
1. Create a **Complex Task Manager Node** that:
   - Implements ROS 2 actions for long-running tasks (e.g., mapping, calibration)
   - Uses multiple callback groups for concurrent processing
   - Manages multiple simultaneous goals
   - Provides feedback during task execution

2. Create a **Task Client Node** that:
   - Sends action goals to the task manager
   - Handles feedback and result processing
   - Manages multiple concurrent tasks

### Implementation Steps
1. Define action message types for the complex tasks
2. Implement the task manager with proper action server implementation
3. Create a client that can send and manage multiple goals
4. Test concurrent task execution and proper resource management

### Expected Outcome
A sophisticated system that demonstrates advanced ROS 2 concepts including actions, callback groups, and concurrent processing.

### Assessment Rubric
- **Action Implementation** (40%): Proper implementation of action server and client
- **Concurrent Processing** (30%): Effective use of callback groups and concurrent task management
- **Resource Management** (20%): Proper handling of multiple concurrent goals and resources
- **Feedback Handling** (10%): Effective use of action feedback mechanisms

## Exercise 4: System Integration and Testing

### Objective
Integrate all concepts learned in Module 1 into a comprehensive system and test it thoroughly.

### Requirements
1. Combine elements from previous exercises into a complete system
2. Implement comprehensive logging and diagnostics
3. Create unit and integration tests for your nodes
4. Document the system architecture and operation

### Implementation Steps
1. Integrate all previous exercises into a cohesive system
2. Add comprehensive logging and diagnostic capabilities
3. Implement tests for each component and the integrated system
4. Create documentation for the complete system
5. Perform system testing and debugging

### Expected Outcome
A complete, well-tested, and documented ROS 2 system that demonstrates mastery of Module 1 concepts.

### Assessment Rubric
- **System Integration** (35%): Successful integration of all components
- **Testing Coverage** (25%): Comprehensive tests for all components
- **Documentation** (20%): Clear and complete documentation
- **Code Quality** (20%): Well-structured, maintainable code following best practices

## Safety Guidelines

- Always test systems in simulation before deploying to physical robots
- Implement proper safety limits and validation checks
- Use appropriate logging to track system behavior
- Ensure proper resource cleanup and error handling
- Validate all inputs and parameter values

## Troubleshooting Tips

1. **Node Communication Issues**: Check topic names, message types, and QoS settings
2. **Performance Problems**: Monitor node execution and consider callback group usage
3. **Parameter Issues**: Verify parameter names, types, and validation logic
4. **Resource Leaks**: Ensure proper cleanup in node destruction

## Further Challenges

For advanced students, consider extending the exercises with:
- Implementing lifecycle nodes for better state management
- Adding security features for protected communication
- Creating custom launch files for system orchestration
- Implementing advanced error handling and recovery mechanisms