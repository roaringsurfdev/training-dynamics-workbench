# REQ_002: Safetensors Model Persistence

## Problem Statement
Current model persistence uses `torch.save()` which relies on pickle format. This has security concerns and version compatibility issues. The mechinterp research community has moved toward safetensors as the standard format.

We need to migrate checkpoint persistence to use safetensors for model weights while maintaining separate storage for training metadata (losses, indices, config).

## Conditions of Satisfaction
- [ ] Model checkpoints saved in safetensors format
- [ ] Training metadata (train_losses, test_losses, train_indices, test_indices, checkpoint_epochs) saved separately (JSON or similar)
- [ ] Model configuration saved in readable format
- [ ] Can load checkpoints from safetensors and reconstruct full training state
- [ ] Backward compatible: can still load old pickle-based checkpoints for analysis

## Constraints
**Must have:**
- Safetensors for all model weights
- TransformerLens compatibility (HookedTransformer can load from safetensors)
- Ability to load individual checkpoints by epoch number

**Must avoid:**
- Pickle format for new checkpoints (security risk)
- Breaking existing trained models (maintain load compatibility)

**Flexible:**
- Metadata format (JSON, YAML, or other readable format)
- Directory structure for organizing checkpoints and metadata
- Whether to store all checkpoints in single file vs. separate files per checkpoint

## Context & Assumptions
- TransformerLens supports safetensors via HuggingFace transformers integration
- Current format: Single pickle file with dict containing model, config, checkpoints, losses, indices
- Safetensors library is mature and well-supported
- Assumption: Individual checkpoint files are acceptable (don't need single monolithic file)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Save training run with safetensors format
- Load checkpoint at specific epoch and verify model weights match
- Load training metadata and verify losses/indices match original
- Attempt to load old pickle checkpoint and verify backward compatibility
- File inspection shows .safetensors extension (not .pth or .pkl)

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
