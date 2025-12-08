---
id: 2
title: Update-robotics-constitution
stage: constitution
date: 2025-12-07
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: User
command: /sp.phr
labels: [constitution, robotics, governance, setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
 - history/prompts/constitution/1-update-project-constitution.constitution.prompt.md
tests:
 - none
---

## Prompt

--stage constitution

/sp.constitution

Project: Physical AI & Humanoid Robotics Course — Panversity Hackathon Edition

Core Principles:
- Deterministic, engineering-grade content; AI-native generation
- Verified robotics knowledge (ROS 2, Isaac Sim, Gazebo, Unity)
- Labs, code, workflows fully reproducible
- Safety and ethics mandatory in humanoid robotics
- Markdown / Docusaurus compatible; CI/CD-ready
- Versioning: Semantic (MAJOR/MINOR/PATCH)

Standards:
- Official docs only: ROS 2, Gazebo, Isaac Sim, Unity
- Code must run/test reproducibly
- Diagrams must be text-renderable
- No hallucinated APIs, commands, frameworks
- Safety disclaimers required for real-world instructions
- Templates deterministic; files Git-friendly

Publishing:
- Docusaurus v3, clean sidebar, search-enabled
- GitHub as source-of-truth; versioned
- Optional: PDF, integrated RAG chatbot

Constraints:
- AI-driven content only
- Real-world humanoid instructions must be ethical/safe
- No chapter/module breakdown here (defined later in /sp.specify)
- Markdown/Docusaurus format required

Scope:
- Physical AI, Humanoid Robotics, VLA, simulation environments
- Modules, lessons, chapters handled later (/sp.specify)
- Governs global rules; content comes later

Success:
- Deterministic templates, reproducible code/labs
- Docusaurus site builds cleanly
- Git repo organized, versioned
- Ethical/safety constraints implemented
- No hallucinated instructions

Version: 1.1.0 | Ratified: 2025-12-07

## Response snapshot

Updated the project constitution file (.specify/memory/constitution.md) with the provided principles for the Physical AI & Humanoid Robotics project. The constitution now includes six core principles: deterministic writing, reproducible code, Markdown+Docusaurus compatibility, Git-friendly structure, strict verifiable outputs, and zero hallucinated APIs. The file includes a sync impact report and governance section with version tracking.

## Outcome

- ✅ Impact: Updated constitution file with project principles and governance
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: None needed
- 🧠 Reflection: Constitution now properly configured for the robotics project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A