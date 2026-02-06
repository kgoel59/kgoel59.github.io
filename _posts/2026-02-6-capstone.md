---
layout: post  
title: "Irrational Generative Agents – Capstone Project"  
date: 2025-02-06 15:09:00  
description: "A Cognitive Agent Framework for Simulating Human-Like Irrational Decision-Making"  
tags: projects learning uow  
categories: projects  
giscus_comments: true  
featured: false  
---

[Project Report](/assets/pdf/capstone/998_report.pdf)

[DEMO](https://www.awesomescreenshot.com/video/39801301?key=6ce095381ebd227570f3c39689545145)

## Project Overview

This capstone project presents **Irrational Generative Agents**, a cognitive agent framework designed to simulate **human-like reasoning and decision-making**, including both **rational and irrational behaviors**.

The system aims to bridge gaps between psychology, artificial intelligence, and agent-based simulation by modeling memory, emotion, motivation, and cognitive bias within autonomous agents operating in a simulated world.

---

## Background and Motivation

Modeling realistic human behavior remains a core challenge in both psychology and AI.

Key challenges motivating this project include:
- Human-subject research is **costly, slow, and ethically constrained**
- Existing virtual agents lack **deep psychological grounding**
- Most agents fail to model **persistent irrational biases** or memory-driven behavior

This project addresses these limitations by creating agents capable of **bounded rationality**, emotional influence, and long-term behavioral evolution.

---

## Framework Overview

The framework integrates **dual-process cognitive theory**:
- **System 1:** Fast, intuitive, heuristic-driven reasoning
- **System 2:** Slow, analytical, deliberative reasoning

### Core Characteristics
- Python-based cognitive and decision-making model
- Unity-powered 2D simulation environment
- Customizable scenarios with real-time observation
- Modular architecture supporting experimentation and extension

---

## Key Contributions

- Design of an **irrational agent architecture** integrating:
  - Sensory input
  - Motivation
  - Emotion
  - Cognitive biases
- A **modular framework** for studying bias-driven decision-making
- A **lifelike agent loop** enabling long-term behavioral growth
- A visually grounded simulation world for real-time analysis

---

## Project Management and Structure

- Agile **Scrum methodology**
- Project duration: **August 2024 – May 2025**
- Three development sprints:
  1. Exploration & Design
  2. Coding & Development
  3. Validation & Testing

Each sprint focused on clearly defined deliverables to ensure steady and measurable progress.

---

## Requirements and Features

### Core Requirements
- Support for both **System 1 and System 2** reasoning
- Simulation of **psychological biases**
- Persistent memory and personality modeling

### Implemented Features
- 2D town simulation environment
- Agent personality and emotion modules
- Social interaction dynamics

### Excluded (v1.0)
- Dynamic map generation
- Real-time direct user control of agents

---

## Use Cases and Team Collaboration

### Primary Use Cases
- Simulation initialization
- Autonomous agent behavior cycles
- Observation data export

### Team Workflow
- Weekly team meetings
- Biweekly supervisor check-ins
- Tooling:
  - GitHub (version control)
  - Slack (communication)
  - OneDrive (documentation)

---

## Methodology and Literature Review

The framework draws inspiration from:
- Generative Agents
- Humanoid Agents
- Evolving Agent systems

Addressed limitations include:
- Lack of persistent irrational biases
- Weak grounding of internal psychological states

### Prompting Strategy
- Combination of **zero-shot** and **few-shot** prompting
- Structured prompt templates to stabilize LLM outputs

---

## Memory and Decision-Making Logic

### Memory System
- **Short-term memory:** recent events and interactions
- **Long-term memory:** consolidated knowledge
- Memories are managed using an **impact index**

### Decision-Making Flow
- System 1 handles quick, low-cost decisions
- System 2 manages planning, evaluation, and conflict resolution
- Memory influences reasoning depth and behavioral consistency

---

## Evaluation Strategy

Evaluation compares **agent decisions** against **human participants**.

### Metrics Used
- **Kullback–Leibler Divergence** for decision distribution similarity
- Parallel decision-making tasks for agents and humans
- Scenario-based behavioral comparison

---

## Study Metrics for Agent Behavior

- Human-choice prediction accuracy
- Five personality dimensions scored from **0–10**
- Behavioral maturity:
  - Low
  - Medium
  - High
- Evaluation of **n = 7** diverse personality profiles
- Bias-awareness isolation through controlled prompting

---

## System Architecture Overview

### Business Architecture
- Task configuration → cognitive processing → evaluation

### Application Architecture
- Perception module
- Planning module
- Bias evaluation module
- Memory module
- Action execution module

Agents are autonomous entities with personality, emotion, memory, and goals.

---

## Agent Internal Architecture

Key components:
- Stimulus processing
- Dual-system decision-making
- Action execution pipeline

### Modules
- **Plan Module:** generates candidate plans
- **Evaluation Module:** scores plans using bias and memory relevance
- **Memory Module:** ensures behavioral consistency over time

---

## Cognitive Processing Pipeline

- Environmental perception and interpretation
- Semantic compression for memory and reasoning
- Dynamic switching between System 1 and System 2
- Context-aware behavioral variation

---

## Inter-Agent Communication

Agents communicate **indirectly** via the environment:
- Map tiles store interaction traces
- Agents observe tiles within perceptual radius
- Enables emergent social behavior

Interaction types include:
- Entity perception
- NPC interactions
- Environmental reminders

---

## Daily Behavior Cycle

1. Agent initialization (personality & emotion)
2. Environmental perception
3. Intent generation
4. Action planning and execution
5. End-of-day memory consolidation and trait adjustment

---

## Action Planning Cycle

- Inputs:
  - Environmental perception
  - Daily goals
  - Internal emotional state
- Plan categories:
  - THINK
  - CHAT
  - MOVE
  - INTERACT
- Probabilistic plan selection
- Post-action internal state updates

---

## Frontend Architecture

- **React:** UI, menus, HUD
- **Phaser:** 2D simulation and interaction
- **Socket.IO:** real-time frontend–backend communication
- Supports multiplayer state synchronization

---

## Technology Stack

### Frontend
- React
- Phaser

### Backend
- Python
- LangSmith (LLM integration)

### Communication
- Socket.IO

The system is modular and designed for scalability.

---

## Evaluation Metrics

- KL Divergence
- Top-k behavioral overlap
- Big Five personality shift
- Behavioral maturity indicators:
  - Multi-step planning
  - Emotional regulation consistency

---

## Experimental Setup

- Web-based sandbox environment
- 26 agents with distinct personalities
- Full logging of actions, emotions, and decisions
- Bias-focused study using a subset of 3 agents
- Human participants provide baseline comparisons

---

## Results and Analysis

Key findings:
- Mixed System 1 + System 2 reasoning aligns best with human behavior
- Personality traits remain mostly stable
- Behavioral maturity varies across agents
- Emotional regulation improves over time, with occasional bias spikes

---

## Modular Bias Evaluation

- Clear behavioral differences between biased and unbiased agents
- Bias-aware agents:
  - Show richer narratives
  - Engage in deeper conversations
  - Exhibit stronger emotional tone
- Bias significantly impacts social dynamics and decision-making

---

## Limitations and Future Work

### Current Limitations
- No self-calibration of goal–behavior alignment
- Static heuristics and personality filters
- Fixed perceptual and interaction radii

### Future Directions
- Adaptive self-evaluation mechanisms
- Dynamic personality evolution
- More realistic social modeling
- Systematic large-scale evaluation

---

## Conclusion

The **Irrational Generative Agents** framework successfully simulates **bounded and irrational human decision-making**.

- Agents exhibit diverse, personality-driven behavior
- Memory integration enables long-term evolution
- Emotional volatility decreases over time
- The architecture provides a strong foundation for future research into cognition and behavior
