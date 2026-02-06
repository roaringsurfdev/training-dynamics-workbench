# [Project Name]
Training Dynamics Workbench

## Summary
The workbench exists to answer: **"How does behavior X change across training?"**

For that question to be meaningful, analysis must hold constant:
- The trained model instance (Model Variant)
- The probe dataset

The **only independent variable** is the checkpoint (training moment).

This invariant is what the platform enforces. Without it, researcher error could introduce confounding variables—different probes, accidental model variations—making visualizations misleading. The workbench systematizes this so visualizations are scientifically meaningful.

## Purpose
The primary purpose of this project is to systematize the training, analysis, and visualization of models over the course of their training.

Unlike typical ML training platforms, this platform is focused on studying *how* models learn a task, not on *optimizing* a model for a given task.

For any given model, this platform is designed to allow a mechanistic researcher to define a Model Family, train Variants of that family, define Probes, and to consistently apply a given Probe across custom-defined Training Checkpoints for single Model Variant.

By applying a single Probe across Variant Training Checkpoints systematically, a researcher will be able to reliable study emergence across training knowing that the only independent variable is the Training Checkpoint.

The secondary purpose, but still central to the purpose of the project, is to create a streamlined and parallelizable analysis engine that allows a researcher to define analysis data they want to capture over training and to define visualizations that are most meaningful for a given Model Family. This allows a researcher to focus on exploring visualizations for signs of emergent behavior.

The analysis engine should be available through a notebook for exploratory visualization design.

The workbench provides a web-based dashboard interface for exploring visualizations across training. The researcher should be able to choose a given Model Family, a subset of analyses and visualizations, and compile data results using a Variant across its Training Checkpoints.
This platform should allow the researcher to spend more time doing analysis and designing visualizations and to minimize time spend training models and generating analysis data.

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

### Key Concepts & Terminology

**Model Family:** A declared grouping of models that share architecture, valid analyzers, and visualizations. Examples: "Modulo Addition 1-Layer", "Indirect Object Identification". Families are defined by `family.json` files in `model_families/`. The family is responsible for creating models and definining analysis and visualization sets that are useful across variants. Families are explicitly registered because what constitutes "structurally similar" is learned over time by the researcher. The Model Family defines the training datasets and the Probe datasets. Training and Probe datasets should be able to accomodate variations in domain parameters that generate Variants. By making the model responsible for training and probe data, this ensures that data generation remains constant across variants, which makes it easier to attribute differences in emergent behavior to changes in parameters as opposed to errors or discrepancies in data generation.

**Model Variant:** A specific trained model within a family. Variants share architecture and analysis logic but differ in domain parameters. In the "Modulo Addition 1-Layer" example, a Model Variant would be a model trainined on a different Modulus or Seed value without changing any of the model architecture. Model Variants are meant to allow researchers to explore how small changes to task definitions and seed values affect training dynamics. Each variant contains its own checkpoints and analysis artifacts directories. Each variant contains its own list of Probe datasets.

**Probe:** The input data used by a Model Family during analysis forward passes. For small toy models, this might be one canonical dataset (e.g., full (a, b) grid for Modulo Addition). For larger models, a Model Family may contain many smaller probes that exercise behaviors of interest. Probe design is part of the research for larger models.

**Checkpoint:** A snapshot of model weights at a specific training epoch. The workbench saves checkpoints at configurable intervals to enable analysis of how behaviors emerge over training.

**Analyzer:**
A module responsible for generating analysis data given a Model Variant Checkpoint and its activation cache. Computes a single analysis function and returns numpy arrays as analysis artifacts. It's possible to re-use Analyzers across multiple Model Families.

**Analysis Run:** 
The workbench focuses on analysis runs instead of training runs. The goal is to optimize the ability to analyze models across training checkpoints instead of optimizing models themselves. Analysis Runs orchestrate the creation of analysis dataset artifacts. The Analysis Run is reponsible for loading checkpoints of a Model Variant, executing forward passes through each checkpoint, passing the output of the forward pass and activation cache to each Analyzer defined in the run. (Note: within the codebase, the Analysis Run is called AnalysisPipeline. I'm intentionally keeping this discrepancy for now.)

**Analysis Report:**
A web-based report made up of visualization components. A single Analysis Report and its Visualization components can be used by any Variant within a Model Family. The data rendered by the visualizers is generated from Analyzers generating Analysis artifacts on a Model Variant's training checkpoints.

## High-Level Architecture
Training Runner: 
- Responsible for executing training runs
- Models limited to what is supported by TransformerLens
- Ideally, model configuration files + training data modules should be configurable. Training data modules may need to be code modules for generating synthetic data
- Responsible for creating model checkpoints
- Initially, model checkpoints will be accessible to the runner as an array of important checkpoints. Going forward, it may be possible to create checkpoints programmatically based on deterministic model training behavior. (EX: change in TEST LOSS curve might kick off higher checkpoint rate)

Analysis Engine:
- Responsible for loading checkpoints, executing forward passes with probes, and generating analysis artifacts
- Enforces the scientific invariant: same Variant + same Probe across all checkpoints
- Receives work via AnalysisPipelineConfig (which analyzers, which checkpoints)
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

**v0.2.0 — First Foundational Release**

This release takes the project from prototype to a foundational architecture. The MVP proved viability; v0.2.0 establishes the abstractions and infrastructure for sustained research.

**Completed (v0.2.0):**
- Model Family abstraction (REQ_021a): ModelFamily protocol, Variant class, FamilyRegistry
- Analysis library architecture (REQ_021b): library/ + analyzers/ separation
- Modulo Addition 1-Layer family implementation (REQ_021c)
- Dashboard integration with family-aware Analysis tab (REQ_021d)
- Training integration with family selection (REQ_021e)
- Per-epoch artifact storage (REQ_021f): eliminates memory exhaustion, enables on-demand loading
- End-to-end workflow: Family selection, variant training, analysis, visualization
- Five trained variants across different primes (p=97, 101, 103, 109, 113)

**Completed (v0.1.x — MVP):**
- First pass at end-to-end process: Training -> Analysis -> Visualizations
- Three core visualizations (Dominant Frequencies, Neuron Activations, Frequency Clusters)
- Interactive slider for navigating visualizations in sync with loss curves
- Gradio dashboard with Training and Analysis tabs
- Project structure and collaboration framework

**Next Up:**
- Notebook-based exploratory visualization design
- Gap-filling pattern for incremental analysis
- Analysis Report concept (web-based visualization reports per family)

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
- Future: Potential AWS deployment

**Training:**
- Small toy models (Modulo Addition, p=113)
- Local execution, training time not a concern for MVP
- Model specs match Neel Nanda's Grokking Experiment

**Logging:**
- Custom logging for MVP (no external integrations like W&B, MLflow)


## Open Questions
- Optimal visualization presentation for neuron frequency clusters (remove/minimize legend)
- Async architecture patterns for training and analysis

---
**Last Updated:** 2026-02-06