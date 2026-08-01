# SWE Agent Environment

> A lightweight reinforcement learning environment for evaluating and training software engineering LLM agents.

## Overview

SWE Agent Environment is a Gymnasium-inspired environment where language models interact with real software engineering tasks.

An agent receives a coding task, modifies a codebase, executes tests, and receives rewards based on task completion.

## Environment Loop

```
Repository
    ↓
Task / Issue
    ↓
LLM Agent
    ↓
Edit Files
    ↓
Run Tests
    ↓
Reward
    ↓
Next Observation
```

## MVP

- Load a repository
- Present a software engineering task
- Apply agent-generated code changes
- Execute unit tests
- Compute reward
- Return next observation

## Planned Features

- Git sandbox
- Test-based rewards
- Multi-turn episodes
- Docker execution
- LiteLLM integration
- Gymnasium-compatible API

## Tech Stack

- Python
- Gymnasium
- Pytest
- GitPython
- LiteLLM

## Status

🚧 Work in Progress
