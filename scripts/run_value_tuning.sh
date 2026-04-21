#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_value_tuning.sh --train-dataset-root PATH [options]

This script is a small value-only tuning entrypoint:
1. train a value model
2. run value inference on a target dataset
3. export visualization videos

Unlike `run_acp_pipeline.sh`, the inference stage defaults to:
- writing only `complementary_info.value`
- skipping ACP advantage/indicator labeling
- exporting visualization videos only

Required:
  --train-dataset-root PATH            Dataset root used for value training.

Optional:
  --train-dataset-repo-id ID           Default: local/<train-dataset-dirname>
  --infer-dataset-root PATH            Dataset root used for value inference/viz.
                                       Default: same as --train-dataset-root
  --infer-dataset-repo-id ID           Default: local/<infer-dataset-dirname>
  --run-tag TAG                        Default: value_tune_YYYYmmdd_HHMMSS
  --value-gpus SPEC                    GPU ids for value train, e.g. 0 or 0,1. Default: 0
  --infer-gpus SPEC                    GPU ids for value infer, e.g. 0 or 0,1. Default: 0
  --mixed-precision MODE               accelerate mixed precision. Default: bf16
  --value-batch-size N                 Default: 16
  --value-steps N                      Default: 3000
  --value-save-freq N                  Default: 3000
  --value-output-dir PATH              Default: outputs/value_tuning/<run-tag>/value_train
  --value-checkpoint-path PATH         Checkpoint root for inference. Default: --value-output-dir
  --value-checkpoint-ref REF           Default: last
  --infer-batch-size N                 Default: 64
  --infer-output-dir PATH              Default: outputs/value_tuning/<run-tag>/value_infer
  --viz-episodes SPEC                  Default: all
  --viz-video-key KEY                  Default: observation.images.base
  --viz-overwrite                      Overwrite existing viz videos.
  --c-fail-coef X                      Default: 0.995
  --length-scale-quantile X            Default: 0.9
  --freeze-language-model BOOL         Default: true
  --freeze-vision-encoder BOOL         Default: true
  --skip-train                         Skip value training and only run infer/viz.
  --skip-infer                         Skip infer/viz and only run training.
  --help                               Show this message.

Example:
  bash scripts/run_value_tuning.sh \
    --train-dataset-root /root/.cache/huggingface/lerobot/towel_all_merged \
    --train-dataset-repo-id towel_all_merged \
    --infer-dataset-root /mnt/data/dataset/lerobot/lerobot_towel_rl_0416_0_new \
    --infer-dataset-repo-id local/lerobot_towel_rl_0416_0 \
    --value-gpus 0 \
    --infer-gpus 3 \
    --c-fail-coef 0.995 \
    --freeze-language-model true \
    --freeze-vision-encoder true \
    --length-scale-quantile 0.9 \
    --viz-episodes '[0,1,2]'
EOF
}

timestamp_now() {
  date '+%Y%m%d_%H%M%S'
}

default_repo_id_from_root() {
  local dataset_root="$1"
  printf 'local/%s\n' "$(basename "$dataset_root")"
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  "$@"
}

TRAIN_DATASET_ROOT=""
TRAIN_DATASET_REPO_ID=""
INFER_DATASET_ROOT=""
INFER_DATASET_REPO_ID=""

RUN_TAG="value_tune_$(timestamp_now)"
VALUE_GPUS="0"
INFER_GPUS="0"
MIXED_PRECISION="bf16"

VALUE_BATCH_SIZE=16
VALUE_STEPS=3000
VALUE_SAVE_FREQ=3000
VALUE_OUTPUT_DIR=""
VALUE_CHECKPOINT_PATH=""
VALUE_CHECKPOINT_REF="last"

INFER_BATCH_SIZE=64
INFER_OUTPUT_DIR=""
VIZ_EPISODES="all"
VIZ_VIDEO_KEY="observation.images.base"
VIZ_OVERWRITE=0

C_FAIL_COEF=0.995
LENGTH_SCALE_QUANTILE=0.9
FREEZE_LANGUAGE_MODEL=true
FREEZE_VISION_ENCODER=true

SKIP_TRAIN=0
SKIP_INFER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-dataset-root)
      TRAIN_DATASET_ROOT="$2"
      shift 2
      ;;
    --train-dataset-repo-id)
      TRAIN_DATASET_REPO_ID="$2"
      shift 2
      ;;
    --infer-dataset-root)
      INFER_DATASET_ROOT="$2"
      shift 2
      ;;
    --infer-dataset-repo-id)
      INFER_DATASET_REPO_ID="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --value-gpus)
      VALUE_GPUS="$2"
      shift 2
      ;;
    --infer-gpus)
      INFER_GPUS="$2"
      shift 2
      ;;
    --mixed-precision)
      MIXED_PRECISION="$2"
      shift 2
      ;;
    --value-batch-size)
      VALUE_BATCH_SIZE="$2"
      shift 2
      ;;
    --value-steps)
      VALUE_STEPS="$2"
      shift 2
      ;;
    --value-save-freq)
      VALUE_SAVE_FREQ="$2"
      shift 2
      ;;
    --value-output-dir)
      VALUE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --value-checkpoint-path)
      VALUE_CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --value-checkpoint-ref)
      VALUE_CHECKPOINT_REF="$2"
      shift 2
      ;;
    --infer-batch-size)
      INFER_BATCH_SIZE="$2"
      shift 2
      ;;
    --infer-output-dir)
      INFER_OUTPUT_DIR="$2"
      shift 2
      ;;
    --viz-episodes)
      VIZ_EPISODES="$2"
      shift 2
      ;;
    --viz-video-key)
      VIZ_VIDEO_KEY="$2"
      shift 2
      ;;
    --viz-overwrite)
      VIZ_OVERWRITE=1
      shift
      ;;
    --c-fail-coef)
      C_FAIL_COEF="$2"
      shift 2
      ;;
    --length-scale-quantile)
      LENGTH_SCALE_QUANTILE="$2"
      shift 2
      ;;
    --freeze-language-model)
      FREEZE_LANGUAGE_MODEL="$2"
      shift 2
      ;;
    --freeze-vision-encoder)
      FREEZE_VISION_ENCODER="$2"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-infer)
      SKIP_INFER=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$TRAIN_DATASET_ROOT" ]]; then
  echo "--train-dataset-root is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$TRAIN_DATASET_ROOT" ]]; then
  echo "Training dataset root not found: $TRAIN_DATASET_ROOT" >&2
  exit 1
fi

if [[ -z "$TRAIN_DATASET_REPO_ID" ]]; then
  TRAIN_DATASET_REPO_ID="$(default_repo_id_from_root "$TRAIN_DATASET_ROOT")"
fi

if [[ -z "$INFER_DATASET_ROOT" ]]; then
  INFER_DATASET_ROOT="$TRAIN_DATASET_ROOT"
fi

if [[ ! -d "$INFER_DATASET_ROOT" ]]; then
  echo "Inference dataset root not found: $INFER_DATASET_ROOT" >&2
  exit 1
fi

if [[ -z "$INFER_DATASET_REPO_ID" ]]; then
  INFER_DATASET_REPO_ID="$(default_repo_id_from_root "$INFER_DATASET_ROOT")"
fi

OUTPUT_BASE="outputs/value_tuning/${RUN_TAG}"
VALUE_OUTPUT_DIR="${VALUE_OUTPUT_DIR:-${OUTPUT_BASE}/value_train}"
VALUE_CHECKPOINT_PATH="${VALUE_CHECKPOINT_PATH:-$VALUE_OUTPUT_DIR}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-${OUTPUT_BASE}/value_infer}"

echo "Value tuning config:"
echo "  run_tag:                 $RUN_TAG"
echo "  train_dataset_root:      $TRAIN_DATASET_ROOT"
echo "  train_dataset_repo_id:   $TRAIN_DATASET_REPO_ID"
echo "  infer_dataset_root:      $INFER_DATASET_ROOT"
echo "  infer_dataset_repo_id:   $INFER_DATASET_REPO_ID"
echo "  value_output_dir:        $VALUE_OUTPUT_DIR"
echo "  value_checkpoint_path:   $VALUE_CHECKPOINT_PATH"
echo "  infer_output_dir:        $INFER_OUTPUT_DIR"
echo "  c_fail_coef:             $C_FAIL_COEF"
echo "  freeze_language_model:   $FREEZE_LANGUAGE_MODEL"
echo "  freeze_vision_encoder:   $FREEZE_VISION_ENCODER"
echo "  length_scale_quantile:   $LENGTH_SCALE_QUANTILE"
echo "  viz_episodes:            $VIZ_EPISODES"
echo "  viz_video_key:           $VIZ_VIDEO_KEY"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  VALUE_TRAIN_ARGS=(
    -m lerobot.scripts.lerobot_value_train
    --value.type=pistar06
    --value.pretrained_path=lerobot/pi05_base
    --value.backbone_source=pi05
    --value.pi05_repo_id=lerobot/pi05_base
    --value.dtype=bfloat16
    --value.push_to_hub=false
    --batch_size="$VALUE_BATCH_SIZE"
    --steps="$VALUE_STEPS"
    --save_freq="$VALUE_SAVE_FREQ"
    --save_checkpoint=true
    --job_name="value_${RUN_TAG}"
    --dataset.repo_id="$TRAIN_DATASET_REPO_ID"
    --dataset.root="$TRAIN_DATASET_ROOT"
    --output_dir="$VALUE_OUTPUT_DIR"
    --targets.c_fail_coef="$C_FAIL_COEF"
    --targets.length_scale_quantile="$LENGTH_SCALE_QUANTILE"
    --value.freeze_language_model="$FREEZE_LANGUAGE_MODEL"
    --value.freeze_vision_encoder="$FREEZE_VISION_ENCODER"
  )

  run_cmd env CUDA_VISIBLE_DEVICES="$VALUE_GPUS" accelerate launch \
    --mixed_precision="$MIXED_PRECISION" \
    "${VALUE_TRAIN_ARGS[@]}"
else
  echo "Skipping value training."
fi

if [[ "$SKIP_INFER" -eq 0 ]]; then
  if [[ "$INFER_GPUS" == "all" ]]; then
    unset CUDA_VISIBLE_DEVICES
    INFER_NUM_PROCESSES="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$INFER_NUM_PROCESSES" -lt 1 ]]; then
      echo "No CUDA devices detected for value inference." >&2
      exit 1
    fi
    INFER_ENV_CMD=(env)
  else
    INFER_NUM_PROCESSES="$(awk -F',' '{print NF}' <<<"$INFER_GPUS")"
    INFER_ENV_CMD=(env "CUDA_VISIBLE_DEVICES=$INFER_GPUS")
  fi

  VALUE_INFER_ARGS=(
    -m lerobot.scripts.lerobot_value_infer
    --dataset.repo_id="$INFER_DATASET_REPO_ID"
    --dataset.root="$INFER_DATASET_ROOT"
    --dataset.download_videos=false
    --inference.checkpoint_path="$VALUE_CHECKPOINT_PATH"
    --inference.checkpoint_ref="$VALUE_CHECKPOINT_REF"
    --runtime.device=cuda
    --runtime.batch_size="$INFER_BATCH_SIZE"
    --acp.enable=false
    --acp.value_field=complementary_info.value
    --viz.enable=true
    --viz.episodes="$VIZ_EPISODES"
    --viz.video_key="$VIZ_VIDEO_KEY"
    --output_dir="$INFER_OUTPUT_DIR"
    --job_name="value_infer_${RUN_TAG}"
  )

  if [[ "$VIZ_OVERWRITE" -eq 1 ]]; then
    VALUE_INFER_ARGS+=(--viz.overwrite=true)
  fi

  if [[ "$INFER_NUM_PROCESSES" -gt 1 ]]; then
    run_cmd "${INFER_ENV_CMD[@]}" accelerate launch \
      --multi_gpu \
      --num_processes="$INFER_NUM_PROCESSES" \
      "${VALUE_INFER_ARGS[@]}"
  else
    run_cmd "${INFER_ENV_CMD[@]}" python \
      "${VALUE_INFER_ARGS[@]}"
  fi
else
  echo "Skipping value inference / visualization."
fi
