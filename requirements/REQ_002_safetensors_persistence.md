# REQ_002: Safetensors Model Persistence

## Problem Statement
Current model persistence uses `torch.save()` which relies on pickle format. This has security concerns and version compatibility issues. The mechinterp research community has moved toward safetensors as the standard format.

Additionally, the current implementation accumulates all checkpoints in memory during training (appending to a list), which wastes memory and doesn't scale well.

We need to migrate checkpoint persistence to use safetensors for model weights, save each checkpoint as a separate file immediately upon creation, and maintain separate storage for training metadata (losses, indices, config).

## Conditions of Satisfaction
- [ ] Each checkpoint saved as separate safetensors file immediately upon creation
- [ ] Checkpoints written to disk during training (not accumulated in memory)
- [ ] Directory structure organizes checkpoints and metadata clearly
- [ ] Training metadata (train_losses, test_losses, train_indices, test_indices) saved separately (JSON or similar)
- [ ] Model configuration saved in readable format
- [ ] Can load individual checkpoint by epoch number
- [ ] Backward compatible: can still load old pickle-based checkpoints for analysis

## Constraints
**Must have:**
- Safetensors for all model weights
- Separate file per checkpoint (each checkpoint is a complete, independently loadable model)
- Immediate disk writes (no accumulation in memory during training)
- TransformerLens compatibility (HookedTransformer can load from safetensors)
- Directory structure following pattern:
  ```
  results/
    {model_name}/
      {model_name}_p{prime}_seed{seed}/
        {model_name}_p(prime)_seed{seed}.safetensors (final model after training run)
        checkpoints/
          checkpoint_epoch_{epoch:05d}.safetensors
        artifacts/        # Analysis artifacts (created by REQ_003)
        metadata.json     # training losses, indices
        config.json       # model configuration
  ```

**Must avoid:**
- Pickle format for new checkpoints (security risk)
- Accumulating checkpoints in memory during training
- Breaking existing trained models (maintain load compatibility)

**Flexible:**
- Exact metadata format (JSON preferred, but YAML or similar acceptable)
- Exact directory naming convention (as long as it's clear and consistent)

## Context & Assumptions
- TransformerLens supports safetensors via HuggingFace transformers integration
- Current format: Single pickle file with dict containing model, config, checkpoints, losses, indices
- Current implementation accumulates checkpoints in memory: `model_checkpoints.append(copy.deepcopy(model.state_dict()))`
- Safetensors library is mature and well-supported
- Each checkpoint represents a complete model state suitable for independent analysis
- Assumption: Separate files enable better memory efficiency and independent checkpoint access

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Save training run with safetensors format
- Verify directory structure matches specification
- Monitor memory usage during training - should remain constant (not grow with checkpoints)
- Load checkpoint at specific epoch independently and verify model weights correct
- Load training metadata and verify losses/indices match original
- Attempt to load old pickle checkpoint and verify backward compatibility
- File inspection shows .safetensors extension (not .pth or .pkl) and proper directory organization

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
