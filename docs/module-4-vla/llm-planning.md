---
sidebar_position: 26
learning_objectives:
  - Understand Large Language Models (LLMs) for cognitive planning in robotics
  - Implement LLM-based reasoning and planning systems for robots
  - Integrate LLMs with multimodal AI systems for enhanced capabilities
  - Create task decomposition and execution planning systems
  - Evaluate LLM performance in robotic planning contexts
prerequisites:
  - Completion of Module 1 (ROS 2 fundamentals)
  - Understanding of multimodal AI concepts (Chapter 3)
  - Basic knowledge of natural language processing
  - Experience with Python and AI frameworks
estimated_time: "4 hours"
---

# Chapter 4: LLM Cognitive Planning and Reasoning

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand how Large Language Models (LLMs) can be used for cognitive planning in robotics
- Implement LLM-based reasoning and planning systems for robotic applications
- Integrate LLMs with multimodal AI systems for enhanced cognitive capabilities
- Create sophisticated task decomposition and execution planning systems
- Evaluate and optimize LLM performance in robotic planning contexts
- Apply best practices for LLM integration in robotics applications

## Introduction

Large Language Models (LLMs) represent a transformative technology for robotic cognitive planning, enabling robots to understand complex natural language instructions, reason about tasks, decompose high-level goals into executable actions, and adapt to novel situations. Unlike traditional rule-based planning systems, LLMs can handle ambiguity, leverage vast world knowledge, and generate flexible plans based on context and common sense reasoning.

LLMs bring several key capabilities to robotic cognitive planning:
- **Natural Language Understanding**: Interpreting complex, natural language commands
- **Common Sense Reasoning**: Applying general world knowledge to specific tasks
- **Task Decomposition**: Breaking down complex goals into manageable subtasks
- **Contextual Adaptation**: Adjusting plans based on environmental context
- **Knowledge Integration**: Leveraging external knowledge bases for planning
- **Flexible Planning**: Generating plans that can adapt to changing conditions

The integration of LLMs with robotics creates opportunities for:
- Natural human-robot interaction through conversation
- Complex task planning without explicit programming
- Transfer of common-sense knowledge to robotic systems
- Adaptive behavior in novel situations
- Collaborative planning with humans

## 1. Theoretical Foundations

### 1.1 LLM Architectures for Planning

**Transformer Architecture**: The foundation for most modern LLMs
- Self-attention mechanisms for understanding context
- Scalability to large amounts of training data
- Sequential processing for planning steps

**Instruction Tuning**: Training models on instruction-following tasks
- Fine-tuning on human feedback (RLHF)
- Chain-of-thought prompting for reasoning
- Few-shot learning capabilities

**Retrieval-Augmented Generation (RAG)**: Combining LLMs with external knowledge
- Integration with robotics knowledge bases
- Context-aware planning using retrieved information
- Factually grounded responses

### 1.2 Cognitive Planning Concepts

**Hierarchical Task Networks (HTNs)**: LLMs can generate hierarchical plans
```
High-level: "Prepare dinner"
├── Subtask: "Find recipe"
├── Subtask: "Gather ingredients"
│   ├── Action: "Go to kitchen"
│   ├── Action: "Identify ingredients"
│   └── Action: "Collect ingredients"
├── Subtask: "Cook meal"
└── Subtask: "Serve food"
```

**Symbolic Planning**: LLMs can interface with symbolic planners
- Converting natural language to formal logic
- Generating PDDL (Planning Domain Definition Language) representations
- Verifying plan feasibility and safety

**Reactive Planning**: Combining LLM reasoning with reactive behaviors
- High-level goal planning with LLMs
- Low-level execution with traditional controllers
- Adaptation based on environmental feedback

### 1.3 Integration with Multimodal Systems

LLMs work synergistically with multimodal systems:
- **Visual Context**: Using vision data to ground language understanding
- **Action Space**: Converting language plans to robotic actions
- **Perception Integration**: Incorporating sensor data into planning
- **Feedback Loops**: Updating plans based on execution results

## 2. Practical Examples

### 2.1 LLM Integration Framework

Implement a framework for integrating LLMs with robotic systems:

```python
# llm_framework.py
import openai
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time

@dataclass
class TaskPlan:
    """Data class for representing a task plan"""
    id: str
    description: str
    subtasks: List['Subtask']
    context: Dict[str, Any]
    created_at: float

@dataclass
class Subtask:
    """Data class for representing a subtask"""
    id: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    priority: int = 0

class LLMCognitivePlanner:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM Cognitive Planner

        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.knowledge_base = {}
        self.robot_capabilities = {
            "navigation": ["move_to", "go_to", "navigate_to"],
            "manipulation": ["grasp", "pickup", "place", "release"],
            "perception": ["detect", "identify", "find", "locate"],
            "communication": ["speak", "listen", "understand"]
        }

    def set_robot_capabilities(self, capabilities: Dict[str, List[str]]):
        """Set the robot's action capabilities"""
        self.robot_capabilities = capabilities

    def add_to_knowledge_base(self, key: str, value: Any):
        """Add information to the knowledge base"""
        self.knowledge_base[key] = value

    def generate_plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> TaskPlan:
        """
        Generate a task plan for a given goal

        Args:
            goal: The high-level goal to achieve
            context: Additional context information

        Returns:
            Generated task plan
        """
        # Prepare the prompt for plan generation
        prompt = self._create_planning_prompt(goal, context)

        # Call the LLM
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse the response
        plan_json = self._extract_json(response.choices[0].message.content)
        plan = self._parse_plan(plan_json)

        return plan

    def _create_planning_prompt(self, goal: str, context: Optional[Dict[str, Any]]) -> str:
        """Create a prompt for task planning"""
        prompt = f"""
        Given the following goal: "{goal}"

        Context information:
        - Robot capabilities: {json.dumps(self.robot_capabilities)}
        - Current environment: {json.dumps(context or {})}
        - Available objects: {json.dumps(self.knowledge_base.get('objects', []))}
        - Current location: {json.dumps(self.knowledge_base.get('location', 'unknown'))}

        Generate a detailed task plan in JSON format with the following structure:
        {{
            "id": "unique_plan_id",
            "description": "brief description of the plan",
            "subtasks": [
                {{
                    "id": "unique_subtask_id",
                    "description": "what to do",
                    "action_type": "one of: navigation, manipulation, perception, communication",
                    "parameters": {{"param1": "value1", ...}},
                    "dependencies": ["subtask_id_1", ...],
                    "priority": integer_priority
                }}
            ],
            "context": {{...}}
        }}

        The plan should be executable by a robot with the specified capabilities.
        Consider the environment and available objects when creating the plan.
        Ensure subtasks are ordered logically and dependencies are respected.
        """

        return prompt

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """
        You are an expert robotic task planner. Your role is to generate detailed, executable task plans for robots based on natural language goals.

        Guidelines:
        1. Break down complex goals into simple, executable subtasks
        2. Use only the robot capabilities provided
        3. Consider environmental constraints and object availability
        4. Ensure subtasks are logically ordered and dependencies are clear
        5. Prioritize safety and feasibility
        6. Provide specific parameters for each action
        7. Output in valid JSON format only
        """

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Look for JSON within code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                json_str = text.strip()

            return json.loads(json_str)
        except json.JSONDecodeError:
            # If direct parsing fails, try to find JSON structure
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"Could not extract JSON from response: {text}")

    def _parse_plan(self, plan_data: Dict[str, Any]) -> TaskPlan:
        """Parse the plan data into a TaskPlan object"""
        subtasks = []
        for subtask_data in plan_data.get("subtasks", []):
            subtask = Subtask(
                id=subtask_data["id"],
                description=subtask_data["description"],
                action_type=subtask_data["action_type"],
                parameters=subtask_data.get("parameters", {}),
                dependencies=subtask_data.get("dependencies", []),
                priority=subtask_data.get("priority", 0)
            )
            subtasks.append(subtask)

        return TaskPlan(
            id=plan_data["id"],
            description=plan_data["description"],
            subtasks=subtasks,
            context=plan_data.get("context", {}),
            created_at=time.time()
        )

    def execute_plan(self, plan: TaskPlan, executor: Callable[[Subtask], bool]) -> Dict[str, Any]:
        """
        Execute a task plan using a provided executor function

        Args:
            plan: The task plan to execute
            executor: Function that executes a subtask and returns success status

        Returns:
            Execution results
        """
        results = {
            "plan_id": plan.id,
            "executed_tasks": [],
            "failed_tasks": [],
            "execution_log": [],
            "success": True
        }

        # Track task execution status
        task_status = {subtask.id: "pending" for subtask in plan.subtasks}

        # Execute tasks in order respecting dependencies
        for subtask in plan.subtasks:
            # Check dependencies
            ready = True
            for dep_id in subtask.dependencies:
                if task_status.get(dep_id) != "completed":
                    ready = False
                    break

            if not ready:
                results["execution_log"].append(f"Skipping {subtask.id}, dependencies not met")
                results["failed_tasks"].append({
                    "subtask": subtask,
                    "reason": "dependencies_not_met"
                })
                results["success"] = False
                continue

            # Execute the task
            try:
                success = executor(subtask)

                if success:
                    task_status[subtask.id] = "completed"
                    results["executed_tasks"].append({
                        "subtask": subtask,
                        "status": "completed"
                    })
                    results["execution_log"].append(f"Completed {subtask.id}: {subtask.description}")
                else:
                    task_status[subtask.id] = "failed"
                    results["failed_tasks"].append({
                        "subtask": subtask,
                        "reason": "executor_failed"
                    })
                    results["success"] = False
                    results["execution_log"].append(f"Failed {subtask.id}: {subtask.description}")

            except Exception as e:
                task_status[subtask.id] = "error"
                results["failed_tasks"].append({
                    "subtask": subtask,
                    "reason": f"exception: {str(e)}"
                })
                results["success"] = False
                results["execution_log"].append(f"Error {subtask.id}: {str(e)}")

        return results
```

### 2.2 Integration with Multimodal Systems

Create integration between LLM planning and multimodal perception:

```python
# llm_multimodal_integration.py
from typing import Dict, Any, Optional
import numpy as np

class MultimodalLLMPlanner:
    def __init__(self, llm_planner: LLMCognitivePlanner, perception_system):
        """
        Integrate LLM planning with multimodal perception

        Args:
            llm_planner: Initialized LLMCognitivePlanner
            perception_system: Multimodal perception system
        """
        self.llm_planner = llm_planner
        self.perception = perception_system

        # Store recent perception results
        self.perception_cache = {}

    def generate_context_aware_plan(self,
                                   goal: str,
                                   image_tensor=None,
                                   audio_transcription: str = "",
                                   current_state: Optional[Dict[str, Any]] = None) -> TaskPlan:
        """
        Generate a plan considering visual and auditory context

        Args:
            goal: The high-level goal
            image_tensor: Current visual input
            audio_transcription: Recent audio input
            current_state: Current robot state

        Returns:
            Context-aware task plan
        """
        # Get multimodal context
        context = self._get_multimodal_context(
            image_tensor, audio_transcription, current_state
        )

        # Generate plan with context
        plan = self.llm_planner.generate_plan(goal, context)

        return plan

    def _get_multimodal_context(self,
                                image_tensor,
                                audio_transcription: str,
                                current_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract context from multimodal inputs"""
        context = {
            "timestamp": time.time(),
            "audio_input": audio_transcription,
            "current_state": current_state or {},
            "objects_detected": [],
            "spatial_relations": [],
            "environment_description": ""
        }

        if image_tensor is not None:
            # Process image through perception system
            with torch.no_grad():
                perception_result = self.perception(image_tensor, [audio_transcription])

            # Extract relevant information
            context["objects_detected"] = self._extract_objects(perception_result)
            context["spatial_relations"] = self._extract_spatial_relations(perception_result)
            context["environment_description"] = self._describe_environment(perception_result)

        return context

    def _extract_objects(self, perception_result) -> List[Dict[str, Any]]:
        """Extract detected objects from perception result"""
        # This would depend on your specific perception system
        # Example implementation:
        objects = []
        if 'object_predictions' in perception_result:
            # Process object detection results
            pass
        return objects

    def _extract_spatial_relations(self, perception_result) -> List[Dict[str, Any]]:
        """Extract spatial relationships from perception result"""
        # This would depend on your specific perception system
        relations = []
        if 'relationship_predictions' in perception_result:
            # Process spatial relationship results
            pass
        return relations

    def _describe_environment(self, perception_result) -> str:
        """Generate a text description of the environment"""
        # This would generate a natural language description
        # based on perception results
        return "Environment description based on visual and audio input"

    def refine_plan_with_feedback(self,
                                 original_plan: TaskPlan,
                                 execution_feedback: Dict[str, Any],
                                 new_context: Optional[Dict[str, Any]] = None) -> TaskPlan:
        """
        Refine a plan based on execution feedback and new context

        Args:
            original_plan: The original task plan
            execution_feedback: Results from plan execution
            new_context: New context information

        Returns:
            Refined task plan
        """
        # Create a refinement prompt
        refinement_prompt = f"""
        Original goal: {original_plan.description}

        Original plan:
        {json.dumps([{
            'id': st.id,
            'description': st.description,
            'action_type': st.action_type,
            'parameters': st.parameters
        } for st in original_plan.subtasks], indent=2)}

        Execution feedback:
        - Executed tasks: {len(execution_feedback.get('executed_tasks', []))}
        - Failed tasks: {len(execution_feedback.get('failed_tasks', []))}
        - Log: {execution_feedback.get('execution_log', [])}

        New context: {json.dumps(new_context or {}, indent=2)}

        Please refine the plan considering the execution feedback and new context.
        If tasks failed, suggest alternative approaches.
        If new information is available, incorporate it into the plan.
        Return the refined plan in the same JSON format.
        """

        # Call the LLM to refine the plan
        response = openai.ChatCompletion.create(
            model=self.llm_planner.model,
            messages=[
                {"role": "system", "content": self.llm_planner._get_system_prompt()},
                {"role": "user", "content": refinement_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Parse the refined plan
        refined_plan_json = self.llm_planner._extract_json(response.choices[0].message.content)
        refined_plan = self.llm_planner._parse_plan(refined_plan_json)

        return refined_plan
```

### 2.3 Task Decomposition and Execution

Implement sophisticated task decomposition and execution systems:

```python
# task_decomposition.py
from enum import Enum
from typing import Union, Tuple

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class AdvancedTaskDecomposer:
    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.task_graph = {}
        self.task_status = {}

    def decompose_complex_task(self,
                              high_level_goal: str,
                              domain_knowledge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Decompose a complex task into multiple levels of subtasks

        Args:
            high_level_goal: The high-level goal to decompose
            domain_knowledge: Specific domain knowledge to guide decomposition

        Returns:
            Dictionary containing the task hierarchy and execution plan
        """
        # First, get a high-level decomposition
        context = {
            "domain_knowledge": domain_knowledge or {},
            "robot_capabilities": self.llm_planner.robot_capabilities
        }

        plan = self.llm_planner.generate_plan(high_level_goal, context)

        # For each subtask, determine if further decomposition is needed
        detailed_plan = self._analyze_and_decompose_subtasks(plan)

        # Create a task execution graph
        execution_graph = self._create_execution_graph(detailed_plan)

        return {
            "original_goal": high_level_goal,
            "decomposed_tasks": detailed_plan,
            "execution_graph": execution_graph,
            "estimated_complexity": self._calculate_complexity(detailed_plan),
            "required_resources": self._identify_resources(detailed_plan)
        }

    def _analyze_and_decompose_subtasks(self, plan: TaskPlan) -> TaskPlan:
        """Analyze each subtask for potential further decomposition"""
        new_subtasks = []

        for subtask in plan.subtasks:
            # Determine if this subtask needs further decomposition
            if self._needs_decomposition(subtask):
                # Ask LLM to decompose this subtask further
                detailed_subtasks = self._decompose_subtask(subtask)
                new_subtasks.extend(detailed_subtasks)
            else:
                new_subtasks.append(subtask)

        plan.subtasks = new_subtasks
        return plan

    def _needs_decomposition(self, subtask: Subtask) -> bool:
        """Determine if a subtask needs further decomposition"""
        # Heuristic: tasks with generic descriptions or complex action types
        # might need further decomposition
        generic_keywords = ["and", "then", "after", "before", "during"]

        # Check if description contains multiple actions or complex logic
        desc_lower = subtask.description.lower()
        if any(keyword in desc_lower for keyword in generic_keywords):
            return True

        # Check if action type is complex
        complex_actions = ["navigate_and_interact", "search_and_manipulate"]
        if subtask.action_type in complex_actions:
            return True

        # Check if parameters are vague
        if not subtask.parameters or len(subtask.parameters) == 0:
            return True

        return False

    def _decompose_subtask(self, subtask: Subtask) -> List[Subtask]:
        """Decompose a subtask into more detailed subtasks"""
        decomposition_prompt = f"""
        Decompose the following subtask into more detailed, executable steps:

        Original subtask:
        - Description: {subtask.description}
        - Action type: {subtask.action_type}
        - Parameters: {subtask.parameters}

        Robot capabilities: {json.dumps(self.llm_planner.robot_capabilities)}

        Decompose this into 2-5 more specific subtasks that can be executed directly.
        Each subtask should have a clear action type and specific parameters.
        Return in the same JSON format as before.
        """

        response = openai.ChatCompletion.create(
            model=self.llm_planner.model,
            messages=[
                {"role": "system", "content": self.llm_planner._get_system_prompt()},
                {"role": "user", "content": decomposition_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )

        try:
            result = self.llm_planner._extract_json(response.choices[0].message.content)
            detailed_subtasks = []

            for subtask_data in result.get("subtasks", []):
                detailed_subtask = Subtask(
                    id=f"{subtask.id}_{subtask_data['id']}",
                    description=subtask_data["description"],
                    action_type=subtask_data["action_type"],
                    parameters=subtask_data.get("parameters", {}),
                    dependencies=[],
                    priority=subtask.priority
                )
                detailed_subtasks.append(detailed_subtask)

            return detailed_subtasks

        except Exception as e:
            print(f"Decomposition failed, returning original: {e}")
            return [subtask]

    def _create_execution_graph(self, plan: TaskPlan) -> Dict[str, Any]:
        """Create an execution graph with dependencies and parallel execution possibilities"""
        graph = {
            "nodes": {},
            "edges": [],
            "parallelizable_groups": [],
            "critical_path": []
        }

        # Build dependency graph
        for subtask in plan.subtasks:
            graph["nodes"][subtask.id] = {
                "subtask": subtask,
                "status": TaskStatus.PENDING,
                "dependencies": subtask.dependencies
            }

        # Identify edges based on dependencies
        for subtask in plan.subtasks:
            for dep_id in subtask.dependencies:
                if dep_id in graph["nodes"]:
                    graph["edges"].append({
                        "from": dep_id,
                        "to": subtask.id
                    })

        # Find parallelizable groups (tasks that can run simultaneously)
        graph["parallelizable_groups"] = self._find_parallelizable_groups(graph)

        # Find critical path (longest sequence of dependent tasks)
        graph["critical_path"] = self._find_critical_path(graph)

        return graph

    def _find_parallelizable_groups(self, graph: Dict[str, Any]) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel"""
        groups = []
        unassigned = set(graph["nodes"].keys())

        while unassigned:
            group = []
            for task_id in list(unassigned):
                # Check if all dependencies are satisfied
                can_execute = True
                for edge in graph["edges"]:
                    if edge["to"] == task_id and edge["from"] in unassigned:
                        can_execute = False
                        break

                if can_execute:
                    group.append(task_id)

            if not group:
                # Deadlock detected - return what we have
                break

            for task_id in group:
                unassigned.remove(task_id)

            groups.append(group)

        return groups

    def _find_critical_path(self, graph: Dict[str, Any]) -> List[str]:
        """Find the critical path through the task graph"""
        # This is a simplified version - in practice, you'd want more sophisticated analysis
        # based on estimated task durations
        visited = set()
        path = []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)

            # Visit dependencies first
            for edge in graph["edges"]:
                if edge["to"] == task_id:
                    dfs(edge["from"])

            path.append(task_id)

        for task_id in graph["nodes"]:
            if task_id not in visited:
                dfs(task_id)

        return path

    def _calculate_complexity(self, plan: TaskPlan) -> int:
        """Calculate the complexity of a plan based on various factors"""
        complexity = len(plan.subtasks)  # Base complexity

        # Add complexity for complex action types
        for subtask in plan.subtasks:
            if subtask.action_type in ["navigation", "manipulation"]:
                complexity += 2
            elif subtask.action_type == "perception":
                complexity += 1

        # Add complexity for dependencies
        for subtask in plan.subtasks:
            complexity += len(subtask.dependencies)

        return complexity

    def _identify_resources(self, plan: TaskPlan) -> Dict[str, int]:
        """Identify required resources for plan execution"""
        resources = {}

        for subtask in plan.subtasks:
            # Count different action types needed
            action_type = subtask.action_type
            resources[action_type] = resources.get(action_type, 0) + 1

            # Identify specific parameters that might require resources
            for param, value in subtask.parameters.items():
                if param in ["location", "object", "tool"]:
                    resource_key = f"{param}_{value}"
                    resources[resource_key] = resources.get(resource_key, 0) + 1

        return resources
```

### 2.4 Integration with ROS 2

Create a ROS 2 node that uses LLM planning:

```python
# llm_planning_ros.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.planner = LLMCognitivePlanner(api_key="your-api-key")
        self.multimodal_planner = MultimodalLLMPlanner(self.planner, perception_system=None)
        self.task_decomposer = AdvancedTaskDecomposer(self.planner)

        # Current state
        self.current_image = None
        self.current_context = {}
        self.active_plan = None

        # Create subscribers
        self.goal_subscriber = self.create_subscription(
            String,
            '/high_level_goal',
            self.goal_callback,
            10
        )

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publishers
        self.plan_publisher = self.create_publisher(
            String,
            '/generated_plan',
            10
        )

        self.action_publisher = self.create_publisher(
            String,
            '/robot_action',
            10
        )

        # Timer for plan execution
        self.execution_timer = self.create_timer(1.0, self.execute_plan_step)

        self.get_logger().info('LLM Planning Node initialized')

    def goal_callback(self, msg):
        """Process high-level goal from user"""
        goal = msg.data
        self.get_logger().info(f'Received goal: {goal}')

        try:
            # Generate plan considering current context
            plan = self.multimodal_planner.generate_context_aware_plan(
                goal=goal,
                image_tensor=self.current_image,
                current_state=self.current_context
            )

            # Decompose complex tasks if needed
            if self._is_complex_goal(goal):
                plan = self.task_decomposer.decompose_complex_task(goal)

            # Store the plan
            self.active_plan = plan

            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'plan_id': plan.id,
                'description': plan.description,
                'subtasks': [
                    {
                        'id': st.id,
                        'description': st.description,
                        'action_type': st.action_type,
                        'parameters': st.parameters
                    } for st in plan.subtasks
                ]
            })
            self.plan_publisher.publish(plan_msg)

            self.get_logger().info(f'Generated plan with {len(plan.subtasks)} subtasks')

        except Exception as e:
            self.get_logger().error(f'Error generating plan: {e}')

    def image_callback(self, msg):
        """Process incoming image data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Convert to tensor format expected by perception system
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def execute_plan_step(self):
        """Execute one step of the active plan"""
        if self.active_plan is None:
            return

        # This is a simplified execution - in practice you'd have a more sophisticated
        # execution manager that tracks progress and handles failures
        for subtask in self.active_plan.subtasks:
            if self._should_execute_subtask(subtask):
                self._execute_subtask(subtask)
                # Mark as completed (in practice, you'd wait for feedback)
                break

    def _should_execute_subtask(self, subtask: Subtask) -> bool:
        """Determine if a subtask should be executed now"""
        # Check dependencies, current state, etc.
        return True  # Simplified for example

    def _execute_subtask(self, subtask: Subtask):
        """Execute a subtask by publishing appropriate commands"""
        action_msg = String()
        action_msg.data = json.dumps({
            'action_type': subtask.action_type,
            'parameters': subtask.parameters,
            'subtask_id': subtask.id
        })
        self.action_publisher.publish(action_msg)

        self.get_logger().info(f'Executing: {subtask.description}')

    def _is_complex_goal(self, goal: str) -> bool:
        """Determine if a goal is complex enough to need decomposition"""
        complex_indicators = [
            "and", "then", "after", "before", "while",
            "navigate to", "pick up", "place", "find and"
        ]
        goal_lower = goal.lower()
        return any(indicator in goal_lower for indicator in complex_indicators)
```

## 3. Hands-on Exercises

### Exercise 1: Basic LLM Integration
**Objective:** Set up and test basic LLM integration for robotic planning.

**Prerequisites:**
- OpenAI API key
- Python with required packages
- Understanding of basic Python programming

**Steps:**
1. Implement the LLMCognitivePlanner class
2. Test with simple robotic goals (e.g., "Move to kitchen")
3. Verify plan generation and JSON parsing
4. Test with different robot capabilities
5. Evaluate plan quality and feasibility

**Expected Outcome:** Working LLM integration that can generate simple task plans for robots.

**Troubleshooting Tips:**
- Ensure API key is valid and has sufficient quota
- Check that prompts are well-formatted
- Verify JSON parsing handles LLM output correctly

### Exercise 2: Multimodal Planning Integration
**Objective:** Integrate LLM planning with multimodal perception systems.

**Prerequisites:**
- Completed Exercise 1
- Understanding of multimodal AI concepts
- Access to perception system or simulated data

**Steps:**
1. Implement the MultimodalLLMPlanner class
2. Integrate with visual perception system
3. Test with context-aware goals (e.g., "Find the red ball and pick it up")
4. Evaluate how well context improves plan quality
5. Test plan refinement based on execution feedback

**Expected Outcome:** System that generates plans considering visual and contextual information.

### Exercise 3: Advanced Task Decomposition
**Objective:** Implement sophisticated task decomposition for complex robotic goals.

**Prerequisites:**
- Completed previous exercises
- Understanding of graph algorithms
- Experience with complex task planning

**Steps:**
1. Implement the AdvancedTaskDecomposer class
2. Test with complex, multi-step goals
3. Evaluate dependency tracking and parallel execution
4. Test plan complexity analysis and resource identification
5. Optimize for real-time performance

**Expected Outcome:** System that can decompose complex goals into executable subtasks with proper dependency management.

## 4. Safety and Ethical Considerations

When implementing LLM-based planning for robotics:
- Ensure LLM-generated plans are validated for safety before execution
- Implement safeguards to prevent unsafe actions from being planned
- Consider bias in LLM training data and its impact on robot behavior
- Plan for graceful degradation when LLM fails or generates invalid plans
- Maintain human oversight for critical planning decisions
- Ensure privacy protection for interactions with LLMs
- Consider the ethical implications of autonomous planning decisions

## 5. Chapter Summary

In this chapter, we've covered:
- The theoretical foundations of LLMs for cognitive planning in robotics
- Practical implementation of LLM-based reasoning and planning systems
- Integration of LLMs with multimodal AI systems for enhanced capabilities
- Advanced task decomposition and execution planning systems
- Integration with ROS 2 for real-world deployment
- Safety and ethical considerations for LLM-based planning

LLM cognitive planning enables robots to understand complex natural language goals, reason about tasks using common sense knowledge, and generate flexible plans that adapt to changing conditions and contexts.

## 6. Assessment Questions

### Multiple Choice
1. What is the primary advantage of using LLMs for robotic planning compared to traditional methods?
   a) Faster execution speed
   b) Ability to handle ambiguity and leverage common sense reasoning
   c) Lower computational requirements
   d) Simpler implementation

   Answer: b) Ability to handle ambiguity and leverage common sense reasoning

2. What does RAG stand for in the context of LLM integration?
   a) Rapid Action Generation
   b) Retrieval-Augmented Generation
   c) Robotic Action Guidance
   d) Reasoning and Generation

   Answer: b) Retrieval-Augmented Generation

### Practical Questions
1. Implement an LLM-based planning system that can decompose complex natural language goals into executable robotic actions, considering visual context and handling plan refinement based on execution feedback.

## 7. Further Reading

- "Language Models for Embodied Intelligence" - Research papers on LLM-robotics integration
- "Chain of Thought Prompting" - Techniques for reasoning with LLMs
- "Retrieval-Augmented Generation for Robotics" - Combining LLMs with knowledge bases
- OpenAI API Documentation: Best practices for LLM integration
- ROS 2 with AI: Integration patterns and best practices