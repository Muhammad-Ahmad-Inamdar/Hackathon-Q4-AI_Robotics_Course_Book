// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/nodes-architecture',
        'module-1-ros2/topics-services',
        'module-1-ros2/rclpy-integration',
        'module-1-ros2/exercises',
        'module-1-ros2/safety',
        'module-1-ros2/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/intro',
        'module-2-digital-twin/gazebo-simulation',
        'module-2-digital-twin/unity-integration',
        'module-2-digital-twin/physics-sensor-simulation',
        'module-2-digital-twin/exercises',
        'module-2-digital-twin/educator-resources',
        'module-2-digital-twin/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-brain/intro',
        'module-3-ai-brain/isaac-sim',
        'module-3-ai-brain/isaac-ros',
        'module-3-ai-brain/nav2-system',
        'module-3-ai-brain/cognitive-systems',
        'module-3-ai-brain/exercises',
        'module-3-ai-brain/summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/fundamentals',
        'module-4-vla/multimodal-ai',
        'module-4-vla/whisper-integration',
        'module-4-vla/llm-planning',
        'module-4-vla/exercises',
        'module-4-vla/summary',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/index',
        'capstone/phase1',
        'capstone/phase2',
        'capstone/phase3',
        'capstone/phase4',
        'capstone/phase5',
        'capstone/phase6',
        'capstone/summary',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/hardware',
        'appendices/safety',
        'appendices/ethics',
        'appendices/glossary',
        'appendices/references',
      ],
    },
  ],
};

export default sidebars;