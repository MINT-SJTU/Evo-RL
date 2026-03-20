# Evo1 Known Issues

## Current scope

- Evo1 in Evo-RL only targets new LeRobot-format checkpoints saved with `config.json`, `model.safetensors`, and processor configs.
- Legacy Evo1 / DeepSpeed checkpoints are intentionally out of scope and are no longer supported by this integration.

## Precision

- Evo1 training is expected to run with `bf16` mixed precision to match the original Evo1 recipe.
- Use the dedicated accelerate config in this directory when launching training to keep the precision path aligned.

## Multi-embodiment training

- The model path still accepts `embodiment_id` and `num_categories`, but the current towel dataset does not contain an `embodiment_id` field.
- With the current data, Evo1 training always falls back to `default_embodiment_id=0`.
- Multi-repo multi-robot training is still blocked at the training config layer.
- A full multi-embodiment setup would still need:
  - a stable `embodiment_id` field in the dataset or a deterministic mapping from dataset identity
  - per-robot normalization instead of a single aggregated stats table
