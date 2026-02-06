# Domain Model

This document describes the core domain objects and their relationships in the Training Dynamics Workbench.

## Class Diagram

```mermaid
classDiagram
    direction TB

    class FamilyRegistry {
        -model_families_dir: Path
        -results_dir: Path
        -families: dict[str, ModelFamily]
        +get_family(name) ModelFamily
        +list_families() list[ModelFamily]
        +get_variants(family) list[Variant]
        +create_variant(family, params) Variant
    }

    class ModelFamily {
        <<protocol>>
        +name: str
        +display_name: str
        +config: HookedTransformerConfig
        +domain_parameters: dict[str, ParameterSpec]
        +analyzers: list[str]
        +variant_pattern: str
        +create_model(params, device) HookedTransformer
        +generate_analysis_dataset(params, device) Tensor
        +generate_training_dataset(params, ...) tuple
        +prepare_analysis_context(params, device) dict
        +get_training_config() dict
    }

    class Variant {
        -family: ModelFamily
        -params: dict[str, Any]
        -results_dir: Path
        +name: str
        +state: VariantState
        +variant_dir: Path
        +checkpoints_dir: Path
        +artifacts_dir: Path
        +get_available_checkpoints() list[int]
        +load_checkpoint(epoch) state_dict
        +load_model_at_checkpoint(epoch) HookedTransformer
        +train(...) TrainingResult
    }

    class VariantState {
        <<enumeration>>
        UNTRAINED
        TRAINED
        ANALYZED
    }

    class AnalysisRun {
        -variant: Variant
        -config: AnalysisRunConfig
        -analyzers: list[Analyzer]
        -manifest: dict
        -results: dict
        -state: AnalysisRunState
        +artifacts_dir: str
        +register(analyzer) AnalysisRun
        +run(force, save_every, progress_callback)
        +get_completed_epochs(analyzer_name) list[int]
        +load_artifact(analyzer_name) dict
    }

    class AnalysisRunState {
        <<enumeration>>
        NOT_STARTED
        IN_PROGRESS
        COMPLETE
        TERMINATED
    }

    class AnalysisRunConfig {
        <<dataclass>>
        - family : ModelFamily
        +analyzers: list[str]
        +checkpoints: list[int] | None
    }

    class Analyzer {
        <<protocol>>
        +name: str
        +analyze(model, probe, cache, context) dict[str, ndarray]
    }

    class AnalyzerRegistry {
        -analyzers: dict[str, Analyzer]
        +register(analyzer)
        +get(name) Analyzer
        +get_for_family(family) list[Analyzer]
    }

    class ArtifactLoader {
        +variant: Variant
        +load(analyzer_name) dict
        +list_available() list[str]
    }

    class TrainingResult {
        <<dataclass>>
        +train_losses: list[float]
        +test_losses: list[float]
        +checkpoint_epochs: list[int]
        +final_train_loss: float
        +final_test_loss: float
        +variant_dir: Path
    }

    %% Relationships
    FamilyRegistry "1" --> "*" ModelFamily : discovers
    FamilyRegistry "1" --> "*" Variant : creates
    Variant "*" --> "1" ModelFamily : belongs to
    Variant "1" --> "1" VariantState : has state
    Variant "1" ..> "1" TrainingResult : produces

    AnalysisRun "1" --> "1" Variant : analyzes
    AnalysisRun "1" --> "0..1" AnalysisRunConfig : configured by
    AnalysisRun "1" --> "1" AnalysisRunState : has state
    AnalysisRun "1" --> "*" Analyzer : uses

    AnalyzerRegistry "1" --> "*" Analyzer : manages

    ArtifactLoader "1" --> "1" Variant : loads from

    ModelFamily ..> HookedTransformer : creates
    ModelFamily ..> HookedTransformerConfig : configured by
```

## Core Concepts

### ModelRegistry
Class for creating new and loading existing ModelFamily instances.

### ModelFamily
A **ModelFamily** is a declared grouping of models that share architecture, valid analyzers, and visualizations. Examples: "Modulo Addition 1-Layer", "Indirect Object Identification". Families are defined by `family.json` files in `model_families/`. The family is responsible for creating models and definining analysis and visualization sets that are useful across variants. Families are explicitly registered because what constitutes "structurally similar" is learned over time by the researcher.

### Model Variant
A **Model Variant** is a specific trained model within a family. Variants share architecture and analysis logic but differ in domain parameters. In the "Modulo Addition 1-Layer" example, a Model Variant would be a model trainined on a different Modulus or Seed value without changing any of the model architecture. Model Variants are meant to allow researchers to explore how small changes to task definitions and seed values affect training dynamics. Each variant manages its own checkpoints and analysis artifacts directories. Each variant maintains its own list of Probe datasets.

Variant training results metadata is stored in `results/{model name}/{variant name}/metadata.json`. Variant checkpoints are stored under `results/{model name}/{variant name}/checkpoints/` as `safetensor` files with the name `checkpoint_epoch_{epoch number}.safetensors`. Checkpoint files are saved at each configured checkpoint instead of storing in memory.

### Probe
The input data used by a Model Family during analysis forward passes. For small toy models, this might be one canonical dataset (e.g., full (a, b) grid for Modulo Addition). For larger models, a Model Family may contain many smaller probes that exercise behaviors of interest. Probe design is part of the research for larger models.

### AnalyzerRegistry
Class for creating new and loading existing Analyzer instances.

### Analyzer
A module responsible for generating analysis data given a Model Variant Checkpoint and its activation cache. Computes a single analysis function and returns numpy arrays as analysis artifacts.

Analyzers are defined by 'analyzer.json' in `analyzers/`.

### AnalysisRun
Orchestrates analysis across a model variant's checkpoints given a list of analysis functions. Manages artifact persistence and resumability.
The workbench focuses on analysis runs instead of training runs. The goal is to optimize the ability to analyze models across training checkpoints instead of optimizing models themselves. Analysis Runs orchestrate the creation of analysis dataset artifacts. The Analysis Run is reponsible for loading checkpoints of a Model Variant, executing forward passes through each checkpoint, passing the output of the forward pass and activation cache to each Analyzer defined in the run.

Analyzer results for a given AnalysisRun are stored under `results/{model name}/{variant name}/analysis/` as `npz` files. Completed analyses are logged in `manifest.json` in `results/{model name}/{variant name}/analysis/`.
