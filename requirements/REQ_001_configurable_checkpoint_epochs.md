# REQ_001: Configurable Checkpoint Epochs

## Problem Statement
The training runner currently checkpoints at fixed intervals (every 100 epochs). For analyzing training dynamics, we need fine-grained control over checkpoint timing to capture more snapshots during critical phases (like grokking) and fewer during stable phases.

We need to be able to specify an integer list of epoch numbers where checkpoints should be saved, rather than using a fixed interval.

## Conditions of Satisfaction
- [ ] Training accepts a list of epoch numbers for checkpointing (e.g., [100, 200, 500, 1000, 1500, 2000, ...])
- [ ] Checkpoints are saved only at the specified epochs
- [ ] If checkpoint list is not provided, falls back to reasonable default behavior
- [ ] Checkpoint list can be configured per training run
- [ ] Training loop efficiently handles arbitrary checkpoint spacing

## Constraints
**Must have:**
- Backward compatible with existing ModuloAdditionSpecification interface
- No performance degradation from checkpoint checking logic

**Must avoid:**
- Creating checkpoints at unintended epochs
- Memory issues from holding too many checkpoints simultaneously

**Flexible:**
- How checkpoint list is passed to training (constructor, method parameter, config object)
- Default checkpoint strategy if none specified

## Context & Assumptions
- Current implementation: ModuloAdditionSpecification.train() uses `checkpoint_every = 100`
- Checkpoints stored as list in `model_checkpoints` array
- User wants to densify checkpoints around grokking phase based on prior knowledge of training dynamics
- Assumption: Epoch numbers in list are valid (< num_epochs)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Can train with custom checkpoint list: [100, 500, 1000, 2000, 5000, 10000]
- Verify len(model_checkpoints) == len(checkpoint_list)
- Verify checkpoint_epochs matches provided list
- Training with default behavior still works

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
