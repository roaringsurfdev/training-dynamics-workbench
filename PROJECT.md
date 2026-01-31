# [Project Name]
Training Dynamics Workbench

## Purpose
The goal of this project is to create a system that allows me to train experimental models, export snapshots during training, and use the snapshots for evaluating emergent model behaviors during training.

## Goals
I am able to able analyze mechanistic behaviors that emerge during training.

## Scope
**In scope:**
- This is a project meant to streamline analysis of training dynamics.
- This is meant to provide a workbench for analyzing visualizations of emergent model behaviors in one location. - The goal is to be able to kick off a training run, configure visualizations that might be useful, process snapshots to generate visualizations, and view the final visualizations in one location. 
- It would also be nice to be able visually compare differences in behavior with different parameters.

**Out of scope (for now):**
- This is not meant to be an application for optimizing model performance. 
- This is also not a replacement for Neuropedia.
- This should be limited to smaller toy model analysis (at first)
- No use of cloud infrastructure. All models will be trained locally. This could change.

## Domain Context
[Background knowledge needed to understand the problem space]
I would like to move away from working in Notebooks for doing analysis. Instead, I would like to be able to focus on iterating on useful visualizations for discovering what might be happening during model training.
[Key concepts, terminology, constraints]

## High-Level Architecture
Training Runner: 
- Responsible for executing training runs
- Models limited to what is supported by TransformerLens
- Ideally, model configuration files + training data modules should be configurable. Training data modules may need to be code modules for generating synthetic data
- Responsible for creating model checkpoints
- Initially, model checkpoints will be accessible to the runner as an array of important checkpoints. Going forward, there may be more intelligent decisions to make on when to create checkpoints (EX: change in TEST LOSS curve might kick off higher checkpoint rate)

Analysis Engine:
- Responsible for loading snapshots, executing forward passes, and generating analysis artifacts
- Analysis artifacts might be anything from raw datasets to chart animations

Workbench:
- This is the primary user interface
- Surfaces ability to kick off asynchronous training runs
- Surfaces ability to kick off asynchronous training run analysis via Analysis Engine
- Provides configurable dashboard to view analysis visualizations

## MVP Definition
**Minimum viable product includes:**
Given a single simple toy Transformer model, execute a training run and generate animations for 3 visualizations to be displayed on a simple dashboard that does not accomodate customization.

Toy model: 1-layer Transformer based on the Modulo Addition Grokking Experiment from Neel Nanda. Model definition will be provided in a config file. Training data will be synthetically generated.
3 visualizations from the original notebook: 
    - Dominant embedding frequencies summary. What, if any, frequencies have the most dominant coefficents over the course of training? The visualization should show a line graph of dominant frequencies over the learned embedding space. The line graph should plot ((fourier_bases @ W_E).norm(dim=-1), with fourier_bases=bases defined by the modulus as period, W_E = embedding weights, fourier bases along the x-axis, normed coefficients along the y-axis).
    - Activation Heat Maps. For a given neuron, what activation patterns emerge over training? The heatmap should be an image of (a, b, activation). The model spec calls for 512 mlps, which may be excessive for visualization. Some thought may be required to determine how best to show emergence across neuron activations.
    - TBD

**Success criteria for MVP:**
- [How we'll know MVP is working]
- I can start a training run of the Modulo Addition model.
- I can start the analysis of a training run of the Modulo Addition model
- The dashboard shows 3 visualization that are hard-coded (not configurable)
- I can change a parameter for the Modulo Addition model and see different analysis results
- Parameters available for changing: Modulus(Period), Random Seeds
- [What we'll validate with MVP]
- This will give us a first end-to-end path through the system for future refinements
- We will have a strategy for saving snapshots to be analysed and surfaced on the workbench
- We will learn about storage constraints and areas for future improvements or changes in direction

**Explicitly deferred (post-MVP):**
- [Features that can wait]
- Configuring the workbench visualizations
- Alternate strategies for generating analysis artifacts
- [Optimizations that can come later]
- Optimized asynchronous architecture

## Current Status
**Completed:**
- [What's done]

**In Progress:**
- [What we're working on]

**Next Up:**
- [What's coming]

## Dependencies & Constraints
[External factors, technical limitations, must-use technologies]


## Open Questions
[Things we need to figure out]
[Decisions that need to be made]

---
**Last Updated:** [Date]
```

## Where MVP Lives

**MVP definition should be in PROJECT.md because:**
- It's about project scope and goals, not process
- It provides context for all requirements
- Requirements can reference it ("This is part of MVP" vs "Post-MVP")
- It evolves as you learn (living document)

**How requirements relate to MVP:**
- PROJECT.md defines MVP scope at high level
- Individual requirements can tag themselves as MVP or post-MVP
- This helps me prioritize and understand scope boundaries

## Project Structure With This Addition
```
/
  Claude.md              # Collaboration framework (stable)
  PROJECT.md             # Project-specific context (living)
  /requirements/         # What to build
  /policies/             # How to work
  /notes/                # Observations and ideas