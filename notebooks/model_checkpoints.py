# %% checkpoint definitions
DEFAULT_CHECKPOINT_EPOCHS = sorted(
    list(
        set(
            [
                *range(0, 1500, 100),  # Early training - dense (~15 checkpoints)
                *range(1500, 9000, 500),  # Mid training - moderate (~15 checkpoints)
                *range(9000, 13000, 100),  # Grokking region - dense (~100 checkpoints)
                *range(13000, 25000, 500),  # Post-grokking - moderate (~24 checkpoints)
            ]
        )
    )
)
CHECKPOINT_EPOCHS_109_485 = sorted(
    list(
        set(
            [
                *range(0, 6000, 100),  # Early and Mid training combined - dense
                *range(6000, 25000, 500),  # Mid training - moderate
            ]
        )
    )
)

# %% print definition for copy/paste into training run config
print(CHECKPOINT_EPOCHS_109_485)
# %%
