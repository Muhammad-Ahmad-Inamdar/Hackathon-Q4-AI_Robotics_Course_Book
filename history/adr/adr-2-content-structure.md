# ADR-2: Content Structure and Deterministic Approach for AI Textbook

**Status**: Accepted
**Date**: 2025-12-07

## Context

The project requires generating educational content about Physical AI & Humanoid Robotics using AI tools while maintaining technical accuracy and reproducibility. The content must follow deterministic patterns to ensure consistency and verifiability against official documentation.

## Decision

We will structure the content in 6 modules plus a capstone project, with each module following these constraints:

- Maximum 2,000 words per module to ensure digestibility
- Deterministic content based on official documentation only (no hallucinated APIs)
- Git-friendly Markdown format with text-described diagrams
- Safety and ethical guidelines included with each module
- Weekly exercises and assessment rubrics for each module

The modules will be:
1. Module 1: Robotic Nervous System (ROS 2)
2. Module 2: Digital Twin (Gazebo & Unity)
3. Module 3: AI-Robot Brain (NVIDIA Isaac)
4. Module 4: Vision-Language-Action (VLA)
5. Capstone: Integration of all previous modules
6. Appendices: Hardware requirements, safety guidelines, references

## Alternatives Considered

1. **Different module organization**: Organize by complexity level instead of technology stack
   - Pros: May be more intuitive for learning progression
   - Cons: Would fragment technology knowledge, making it harder to follow official documentation

2. **Different content length**: Longer or shorter modules
   - Pros: Shorter modules might be more digestible, longer modules could provide more depth
   - Cons: 2000 words is a good balance between depth and digestibility; longer modules might overwhelm students

3. **Different content approach**: Allow more creative or speculative content instead of strictly deterministic
   - Pros: Might be more engaging or innovative
   - Cons: Would risk technical inaccuracy, hallucinated APIs, and non-reproducible examples

## Consequences

**Positive**:
- Content will be consistent and technically accurate
- Students can reproduce all examples and labs
- Git-friendly format enables proper version control
- Modular structure allows for independent learning paths

**Negative**:
- May be less creative or innovative than speculative approaches
- Strict constraints might limit some pedagogical possibilities
- Requires careful verification against official documentation

## References

- spec.md: Requirements and Success Criteria sections
- plan.md: Constitution Check section