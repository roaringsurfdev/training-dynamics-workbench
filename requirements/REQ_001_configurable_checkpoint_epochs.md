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
- Works with immediate disk writes (REQ_002) - no memory accumulation

**Must avoid:**
- Creating checkpoints at unintended epochs
- Off-by-one errors in epoch numbering

**Flexible:**
- How checkpoint list is passed to training (constructor, method parameter, config object)
- Default checkpoint strategy if none specified

## Context & Assumptions
- Current implementation: ModuloAdditionSpecification.train() uses `checkpoint_every = 100`
- Current implementation accumulates checkpoints in memory (will be replaced by immediate disk writes in REQ_002)
- User wants to densify checkpoints around grokking phase based on prior knowledge of training dynamics
- REQ_002 handles the actual checkpoint persistence (separate files, safetensors)
- This requirement focuses on WHEN to checkpoint, REQ_002 handles HOW to persist
- Assumption: Epoch numbers in list are valid (< num_epochs)
- Assumption: Checkpoint list is sorted (or can be sorted automatically)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Can train with custom checkpoint list: [100, 500, 1000, 2000, 5000, 10000]
- Verify correct number of checkpoint files created on disk
- Verify checkpoint epochs match provided list exactly
- Training with default behavior still works (reasonable default checkpoint schedule)
- No checkpoints created at unspecified epochs

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
