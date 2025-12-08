# ADR-1: Technology Stack and Architecture for Physical AI & Humanoid Robotics Textbook

**Status**: Accepted
**Date**: 2025-12-07

## Context

The project requires building an AI-native textbook on Physical AI & Humanoid Robotics with integrated RAG chatbot functionality. The system needs to support educational content delivery, interactive Q&A based on textbook content, and integration with multiple robotics frameworks (ROS 2, Gazebo, Unity, NVIDIA Isaac).

## Decision

We will use the following technology stack:

**Frontend/Presentation Layer**:
- Docusaurus v3 for documentation website
- GitHub Pages or Vercel for deployment
- Markdown format for content

**Backend/Chatbot Layer**:
- Python 3.11 with FastAPI for backend API
- OpenAI SDK for AI integration
- Neon Serverless Postgres for relational data
- Qdrant Cloud Free Tier for vector storage

**Target Platforms**:
- ROS 2 (Humble Hawksbill)
- Gazebo for simulation
- Unity for digital twin
- NVIDIA Isaac Sim for AI integration

## Alternatives Considered

1. **Static site generator alternatives**: Hugo, Jekyll, or VuePress instead of Docusaurus
   - Pros: Different ecosystems, potentially different performance characteristics
   - Cons: Less AI/ML documentation focus, less community for technical content

2. **Backend alternatives**: Node.js/Express, Django, or Spring Boot instead of FastAPI
   - Pros: Different language preferences, existing team expertise
   - Cons: Less performant for AI workloads, less async support than FastAPI

3. **Database alternatives**: MongoDB, PostgreSQL (self-hosted), or Supabase instead of Neon + Qdrant
   - Pros: Different data models, single database solution
   - Cons: Vector search capabilities not as optimized, more complex setup

## Consequences

**Positive**:
- Docusaurus is excellent for technical documentation with good search and versioning
- FastAPI provides excellent async performance for AI workloads and has great OpenAPI integration
- The combination of Postgres + Qdrant allows for both structured data and vector similarity search
- ROS 2, Gazebo, Unity, and Isaac are industry standards for robotics education

**Negative**:
- Multiple database systems increase operational complexity
- Multiple technology stacks may require diverse expertise
- Cloud services introduce vendor dependencies and potential costs

## References

- plan.md: Technical Context section
- spec.md: Functional and Non-Functional Requirements