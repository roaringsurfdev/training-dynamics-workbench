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

**Background:** I would like to move away from working in Notebooks for doing analysis. Instead, I would like to be able to focus on iterating on useful visualizations for discovering what might be happening during model training.

### Key Concepts & Terminology

**Model Family:** A declared grouping of models that share architecture, valid analyzers, and visualizations. Examples: "Modulo Addition 1-Layer", "Indirect Object Identification". Families are explicitly registered because what constitutes "structurally similar" is learned over time by the researcher.

**Variant:** A specific trained model within a family. Variants share architecture and analysis logic but differ in domain parameters (e.g., modulus, seed). This is NOT hyperparameter optimization—all variants have solved the problem. The goal is understanding *how* they solve it.

**Probe (Analysis Dataset):** The input data used during analysis forward passes. For small toy models, often one canonical dataset (e.g., full (a, b) grid for Modulo Addition). For larger models, specific probe datasets that exercise behaviors of interest. Probe design is part of the research for larger models.

**Checkpoint:** A snapshot of model weights at a specific training epoch. The workbench saves checkpoints at configurable intervals to enable analysis of how behaviors emerge over training.

### The Scientific Invariant

The workbench exists to answer: **"How does behavior X change across training?"**

For that question to be meaningful, analysis must hold constant:
- The trained model instance (Variant)
- The probe dataset

The **only independent variable** is the checkpoint (training moment).

This invariant is what the platform enforces. Without it, researcher error could introduce confounding variables—different probes, accidental model variations—making visualizations misleading. The workbench systematizes this so visualizations are scientifically meaningful.

## High-Level Architecture
Training Runner: 
- Responsible for executing training runs
- Models limited to what is supported by TransformerLens
- Ideally, model configuration files + training data modules should be configurable. Training data modules may need to be code modules for generating synthetic data
- Responsible for creating model checkpoints
- Initially, model checkpoints will be accessible to the runner as an array of important checkpoints. Going forward, there may be more intelligent decisions to make on when to create checkpoints (EX: change in TEST LOSS curve might kick off higher checkpoint rate)

Analysis Engine:
- Responsible for loading checkpoints, executing forward passes with probes, and generating analysis artifacts
- Enforces the scientific invariant: same Variant + same Probe across all checkpoints
- Receives work via AnalysisRunConfig (which analyzers, which checkpoints)
- Artifacts are keyed by (Variant, Analyzer, Checkpoint) for incremental computation
- Future: gap-filling pattern to compute only missing (analyzer, checkpoint) combinations

Workbench:
- This is the primary user interface
- Surfaces ability to kick off asynchronous training runs
- Surfaces ability to kick off asynchronous training run analysis via Analysis Engine
- Provides configurable dashboard to view analysis visualizations

## MVP (Completed)

MVP was released with v0.1.0. The initial release delivered:
- End-to-end training and analysis of Modulo Addition 1-Layer model
- Three core visualizations (Dominant Frequencies, Neuron Activations, Frequency Clusters)
- Gradio dashboard with basic Training and Analysis tabs
- Parameterization by modulus (p) and seed

See `requirements/archive/` for detailed MVP requirements.

## Current Status
**Completed:**
- Project structure and collaboration framework (Claude.md, policies, requirements templates)
- Baseline modulo addition model implementation (ModuloAdditionSpecification.py)
- Fourier analysis utilities (FourierEvaluation.py)
- Working end-to-end analysis script (ModuloAdditionRefactored.py)
- Parameterized model by modulus (p) with dynamic dominant frequency detection
- Model Family abstraction (REQ_021a): ModelFamily protocol, Variant class, FamilyRegistry
- Analysis library architecture (REQ_021b): library/ + analyzers/ separation
- Modulo Addition 1-Layer family implementation (REQ_021c)
- Dashboard integration with family-aware Analysis tab (REQ_021d)
- Training integration with family selection (REQ_021e - partial)

**In Progress:**
- REQ_021e: Training Integration (end-to-end flow validation)
- Pipeline interface refinement: eliminating VariantSpecificationAdapter

**Next Up:**
- Refactor AnalysisPipeline to take (Variant, AnalysisRunConfig) directly
- Formalize AnalysisRunConfig as a first-class concept
- Gap-filling pattern for incremental analysis

## Dependencies & Constraints

**Technology Stack:**
- Python 3.13
- PyTorch with CUDA support
- TransformerLens (latest stable version)
- Plotly for visualizations
- Gradio for dashboard UI (ML-focused, suitable for interpretability work)
- pytest for testing
- safetensors for model checkpoint persistence
- JAX/Flax (optional, for data generation and analysis computations)

**Checkpoint Strategy:**
- Configurable as integer list of epoch checkpoints
- Allows fine-tuning checkpoint density around grokking phase
- Format: safetensors for model weights, separate metadata for training metrics

**Storage:**
- Local filesystem only for MVP
- 1TB available after WSL instance migration
- Future: Potential AWS deployment and mini rack

**Training:**
- Small toy models (Modulo Addition, p=113)
- Local execution, training time not a concern for MVP
- Model specs match Neel Nanda's Grokking Experiment

**Logging:**
- Custom logging for MVP (no external integrations like W&B, MLflow)


## Open Questions
- Analysis artifact storage format (flexibility for implementation - not critical for MVP)
- Optimal visualization presentation for neuron frequency clusters (remove/minimize legend)
- Post-MVP: Async architecture patterns for training and analysis

---
**Last Updated:** 2026-02-04