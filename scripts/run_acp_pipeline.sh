#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_acp_pipeline.sh --dataset-root /path/to/dataset [options]

This script runs three stages in sequence on a copied dataset:
1. value training on CUDA device 0
2. value inference + ACP annotation writing on CUDA device 0
3. build an ACP training instance table from advantage annotations
4. policy training with ACP on the instance table

Required:
  --dataset-root PATH              Local LeRobot dataset root.

Optional:
  --dataset-repo-id ID             Dataset repo_id passed to LeRobot. Default: local/<dataset-dirname>
  --run-tag TAG                    Shared run tag. Default: acp_YYYYmmdd_HHMMSS
  --field-tag TAG                  ACP field suffix. Default: sanitized --run-tag
  --backup-root PATH               Parent directory for the copied dataset. Default: sibling of dataset root
  --working-dataset-root PATH      Exact copied dataset path to create/use. Overrides --backup-root
  --skip-backup                    Use the original dataset directly. Unsafe.
  --skip-value-train               Skip value training.
  --skip-value-infer               Skip value inference / ACP annotation writing.
  --skip-instance-table-build      Skip ACP instance table generation.
  --skip-policy-train              Skip policy training.
  --value-gpus SPEC                GPU ids visible to value train. Only the first id is used.
                                   Example: 0 or 0,1. Default: 0
  --infer-gpus SPEC                GPUs for value infer. Use 'all' or comma list. Default: 0
  --policy-gpus SPEC               GPUs for policy training. Use 'all' or comma list like 0,1,2,3. Default: all
  --mixed-precision MODE           accelerate mixed precision. Default: bf16
  --value-steps N                  Default: 24000
  --value-batch-size N             Default: 32
  --value-save-freq N              Default: 12000
  --value-output-dir PATH          Default: outputs/pipeline/<run-tag>/value_train
  --value-checkpoint-path PATH     Value checkpoint for inference. Default: --value-output-dir
  --value-checkpoint-ref REF       Default: last
  --value-job-name NAME            Default: value_<run-tag>
  --value-type NAME                Default: pistar06
  --value-dtype NAME               Default: bfloat16
  --infer-batch-size N             Default: 64
  --infer-output-dir PATH          Default: outputs/pipeline/<run-tag>/value_infer
  --infer-job-name NAME            Default: <run-tag>.infer
  --acp-n-step N                   Default: 50
  --acp-positive-ratio X           Default: 0.3
  --c-fail-coef X                  c_fail coefficient for value train/infer. Default: 0.995
  --length-scale-quantile X        Default: 0.9
  --freeze-language-model BOOL     Default: true
  --freeze-vision-encoder BOOL     Default: true
  --acp-instance-table-path PATH   Default: <work-dataset-root>/meta/acp_instance_table_<field-tag>.parquet
  --acp-negative-bottom-ratio X    Default: 0.3
  --acp-positive-top-duplicate-ratio X
                                   Default: 0.3
  --policy-steps N                 Default: 20000
  --policy-batch-size N            Default: 32
  --policy-save-freq N             Default: 5000
  --policy-output-dir PATH         Default: outputs/pipeline/<run-tag>/policy_train
  --policy-resume                  Resume policy training from an existing checkpoint config.
  --policy-config-path PATH        Path to checkpoint pretrained_model/train_config.json.
  --policy-job-name NAME           Default: pi05_<run-tag>
  --policy-type NAME               Default: pi05
  --policy-pretrained-path PATH    Default: lerobot/pi05_base
  --policy-dtype NAME              Default: bfloat16
  --indicator-dropout-prob X       Default: 0.0 (ignored when instance table is used)
  --no-wandb                       Disable wandb for both train stages
  --help                           Show this message

Examples:
  scripts/run_acp_pipeline.sh \
    --dataset-root /mnt/data/dataset/noetix/arx5_subset \
    --dataset-repo-id noetix/arx5_subset \
    --run-tag arx_slipper_v1

  scripts/run_acp_pipeline.sh \
    --dataset-root /mnt/data/dataset/noetix/arx5_subset \
    --skip-value-train \
    --value-checkpoint-path outputs/pipeline/old_run/value_train

  scripts/run_acp_pipeline.sh \
    --dataset-root /mnt/data/dataset/noetix/arx5_subset \
    --skip-value-train \
    --skip-value-infer

  scripts/run_acp_pipeline.sh \
    --dataset-root /mnt/data/dataset/noetix/arx5_subset \
    --skip-backup \
    --skip-value-train \
    --skip-value-infer \
    --policy-resume \
    --policy-config-path outputs/pipeline/old_run/policy_train/checkpoints/005000/pretrained_model/train_config.json
EOF
}

timestamp_now() {
  date '+%Y%m%d_%H%M%S'
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

sanitize_tag() {
  local raw="$1"
  local cleaned
  cleaned="$(printf '%s' "$raw" | tr -cs 'A-Za-z0-9_' '_')"
  cleaned="${cleaned#_}"
  cleaned="${cleaned%_}"
  if [[ -z "$cleaned" ]]; then
    cleaned="tag_$(timestamp_now)"
  fi
  printf '%s\n' "$cleaned"
}

activate_repo_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda activate evo-rl; then
    return
  fi

  conda activate /llm_jzm/cache/conda_env/lerobot
  export HF_HOME=/llm_jzm/cache/huggingface/
  export HF_ENDPOINT=https://hf-mirror.com
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
  fi
}

copy_dataset() {
  local src="$1"
  local dst="$2"

  if [[ -e "$dst" ]]; then
    echo "Refusing to overwrite existing backup dataset: $dst" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dst")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -aH "$src"/ "$dst"/
  else
    cp -a "$src" "$dst"
  fi
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

first_gpu_from_spec() {
  local gpu_spec="$1"
  printf '%s\n' "${gpu_spec%%,*}"
}

run_cmd() {
  print_cmd "$@"
  "$@"
}

DATASET_ROOT=""
DATASET_REPO_ID=""
RUN_TAG="acp_$(timestamp_now)"
FIELD_TAG=""
BACKUP_ROOT=""
WORKING_DATASET_ROOT=""
SKIP_BACKUP=0
SKIP_VALUE_TRAIN=0
SKIP_VALUE_INFER=0
SKIP_INSTANCE_TABLE_BUILD=0
SKIP_POLICY_TRAIN=0
POLICY_RESUME=0

VALUE_GPUS="0"
INFER_GPUS="0"
POLICY_GPUS="all"
MIXED_PRECISION="bf16"

VALUE_STEPS=24000
VALUE_BATCH_SIZE=32
VALUE_SAVE_FREQ=12000
VALUE_CHECKPOINT_PATH=""
VALUE_CHECKPOINT_REF="last"
VALUE_TYPE="pistar06"
VALUE_DTYPE="bfloat16"

INFER_BATCH_SIZE=64
ACP_N_STEP=50
ACP_POSITIVE_RATIO=0.3
C_FAIL_COEF=0.995
LENGTH_SCALE_QUANTILE=0.9
FREEZE_LANGUAGE_MODEL=true
FREEZE_VISION_ENCODER=true
ACP_INSTANCE_TABLE_PATH=""
ACP_NEGATIVE_BOTTOM_RATIO=0.3
ACP_POSITIVE_TOP_DUPLICATE_RATIO=0.3

POLICY_STEPS=20000
POLICY_BATCH_SIZE=32
POLICY_SAVE_FREQ=5000
POLICY_CONFIG_PATH=""
POLICY_TYPE="pi05"
POLICY_PRETRAINED_PATH="lerobot/pi05_base"
POLICY_DTYPE="bfloat16"
INDICATOR_DROPOUT_PROB=0.0

WANDB_ENABLE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset-repo-id)
      DATASET_REPO_ID="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --field-tag)
      FIELD_TAG="$2"
      shift 2
      ;;
    --backup-root)
      BACKUP_ROOT="$2"
      shift 2
      ;;
    --working-dataset-root)
      WORKING_DATASET_ROOT="$2"
      shift 2
      ;;
    --skip-backup)
      SKIP_BACKUP=1
      shift
      ;;
    --skip-value-train)
      SKIP_VALUE_TRAIN=1
      shift
      ;;
    --skip-value-infer)
      SKIP_VALUE_INFER=1
      shift
      ;;
    --skip-instance-table-build)
      SKIP_INSTANCE_TABLE_BUILD=1
      shift
      ;;
    --skip-policy-train)
      SKIP_POLICY_TRAIN=1
      shift
      ;;
    --policy-resume)
      POLICY_RESUME=1
      shift
      ;;
    --value-gpus)
      VALUE_GPUS="$2"
      shift 2
      ;;
    --value-gpu)
      VALUE_GPUS="$2"
      shift 2
      ;;
    --infer-gpus)
      INFER_GPUS="$2"
      shift 2
      ;;
    --policy-gpus)
      POLICY_GPUS="$2"
      shift 2
      ;;
    --mixed-precision)
      MIXED_PRECISION="$2"
      shift 2
      ;;
    --value-steps)
      VALUE_STEPS="$2"
      shift 2
      ;;
    --value-batch-size)
      VALUE_BATCH_SIZE="$2"
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
    --value-job-name)
      VALUE_JOB_NAME="$2"
      shift 2
      ;;
    --value-type)
      VALUE_TYPE="$2"
      shift 2
      ;;
    --value-dtype)
      VALUE_DTYPE="$2"
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
    --infer-job-name)
      INFER_JOB_NAME="$2"
      shift 2
      ;;
    --acp-n-step)
      ACP_N_STEP="$2"
      shift 2
      ;;
    --acp-positive-ratio)
      ACP_POSITIVE_RATIO="$2"
      shift 2
      ;;
    --c_fail_coef)
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
    --acp-instance-table-path)
      ACP_INSTANCE_TABLE_PATH="$2"
      shift 2
      ;;
    --acp-negative-bottom-ratio)
      ACP_NEGATIVE_BOTTOM_RATIO="$2"
      shift 2
      ;;
    --acp-positive-top-duplicate-ratio)
      ACP_POSITIVE_TOP_DUPLICATE_RATIO="$2"
      shift 2
      ;;
    --policy-steps)
      POLICY_STEPS="$2"
      shift 2
      ;;
    --policy-batch-size)
      POLICY_BATCH_SIZE="$2"
      shift 2
      ;;
    --policy-save-freq)
      POLICY_SAVE_FREQ="$2"
      shift 2
      ;;
    --policy-output-dir)
      POLICY_OUTPUT_DIR="$2"
      shift 2
      ;;
    --policy-config-path)
      POLICY_CONFIG_PATH="$2"
      shift 2
      ;;
    --policy-job-name)
      POLICY_JOB_NAME="$2"
      shift 2
      ;;
    --policy-type)
      POLICY_TYPE="$2"
      shift 2
      ;;
    --policy-pretrained-path)
      POLICY_PRETRAINED_PATH="$2"
      shift 2
      ;;
    --policy-dtype)
      POLICY_DTYPE="$2"
      shift 2
      ;;
    --indicator-dropout-prob)
      INDICATOR_DROPOUT_PROB="$2"
      shift 2
      ;;
    --no-wandb)
      WANDB_ENABLE=false
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "--dataset-root is required." >&2
  usage
  exit 1
fi

DATASET_ROOT="$(realpath "$DATASET_ROOT")"
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi

if [[ -z "$DATASET_REPO_ID" ]]; then
  DATASET_REPO_ID="local/$(basename "$DATASET_ROOT")"
fi

FIELD_TAG="$(sanitize_tag "${FIELD_TAG:-$RUN_TAG}")"

OUTPUT_BASE="outputs/pipeline/${RUN_TAG}"
VALUE_OUTPUT_DIR="${VALUE_OUTPUT_DIR:-${OUTPUT_BASE}/value_train}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-${OUTPUT_BASE}/value_infer}"
POLICY_OUTPUT_DIR="${POLICY_OUTPUT_DIR:-${OUTPUT_BASE}/policy_train}"
VALUE_CHECKPOINT_PATH="${VALUE_CHECKPOINT_PATH:-$VALUE_OUTPUT_DIR}"
VALUE_TRAIN_GPU="$(first_gpu_from_spec "$VALUE_GPUS")"

VALUE_JOB_NAME="${VALUE_JOB_NAME:-value_${RUN_TAG}}"
INFER_JOB_NAME="${INFER_JOB_NAME:-${RUN_TAG}.infer}"
POLICY_JOB_NAME="${POLICY_JOB_NAME:-pi05_${RUN_TAG}}"

VALUE_FIELD="complementary_info.value_${FIELD_TAG}"
ADVANTAGE_FIELD="complementary_info.advantage_${FIELD_TAG}"
INDICATOR_FIELD="complementary_info.acp_indicator_${FIELD_TAG}"

if [[ "$SKIP_BACKUP" -eq 1 ]]; then
  WORK_DATASET_ROOT="$DATASET_ROOT"
else
  if [[ -z "$WORKING_DATASET_ROOT" ]]; then
    if [[ -z "$BACKUP_ROOT" ]]; then
      BACKUP_ROOT="$(dirname "$DATASET_ROOT")"
    fi
    WORKING_DATASET_ROOT="${BACKUP_ROOT}/$(basename "$DATASET_ROOT")_acp_${RUN_TAG}"
  fi
  WORK_DATASET_ROOT="$WORKING_DATASET_ROOT"
fi

ACP_INSTANCE_TABLE_PATH="${ACP_INSTANCE_TABLE_PATH:-${WORK_DATASET_ROOT}/meta/acp_instance_table_${FIELD_TAG}.parquet}"

# activate_repo_conda_env

if [[ "$SKIP_BACKUP" -eq 0 ]]; then
  echo "Creating dataset backup:"
  echo "  source: $DATASET_ROOT"
  echo "  backup: $WORK_DATASET_ROOT"
  copy_dataset "$DATASET_ROOT" "$WORK_DATASET_ROOT"
else
  echo "Backup skipped. The original dataset will be modified in place:"
  echo "  dataset: $WORK_DATASET_ROOT"
fi

mkdir -p "$OUTPUT_BASE"

if [[ "$SKIP_VALUE_TRAIN" -eq 1 && "$SKIP_VALUE_INFER" -eq 0 ]]; then
  VALUE_CHECKPOINT_PATH="$(realpath "$VALUE_CHECKPOINT_PATH")"
  if [[ ! -e "$VALUE_CHECKPOINT_PATH" ]]; then
    echo "Value checkpoint path does not exist: $VALUE_CHECKPOINT_PATH" >&2
    exit 1
  fi
fi

if [[ "$SKIP_INSTANCE_TABLE_BUILD" -eq 1 && "$SKIP_POLICY_TRAIN" -eq 0 ]]; then
  if [[ ! -f "$ACP_INSTANCE_TABLE_PATH" ]]; then
    echo "ACP instance table path does not exist: $ACP_INSTANCE_TABLE_PATH" >&2
    exit 1
  fi
fi

if [[ "$POLICY_RESUME" -eq 1 ]]; then
  if [[ "$SKIP_POLICY_TRAIN" -eq 1 ]]; then
    echo "--policy-resume cannot be used together with --skip-policy-train." >&2
    exit 1
  fi
  if [[ -z "$POLICY_CONFIG_PATH" ]]; then
    echo "--policy-config-path is required when --policy-resume is set." >&2
    exit 1
  fi
  POLICY_CONFIG_PATH="$(realpath "$POLICY_CONFIG_PATH")"
  if [[ ! -f "$POLICY_CONFIG_PATH" ]]; then
    echo "Policy config path does not exist: $POLICY_CONFIG_PATH" >&2
    exit 1
  fi
  if [[ "$(basename "$POLICY_CONFIG_PATH")" != "train_config.json" ]]; then
    echo "--policy-config-path should point to a train_config.json file." >&2
    exit 1
  fi
fi

if [[ "$SKIP_VALUE_TRAIN" -eq 1 && "$SKIP_VALUE_INFER" -eq 1 && "$SKIP_INSTANCE_TABLE_BUILD" -eq 1 && "$SKIP_POLICY_TRAIN" -eq 1 ]]; then
  echo "Nothing to do: all stages are skipped." >&2
  exit 1
fi

OUTPUT_PATHS_TO_CHECK=()
if [[ "$SKIP_VALUE_TRAIN" -eq 0 ]]; then
  OUTPUT_PATHS_TO_CHECK+=("$VALUE_OUTPUT_DIR")
fi
if [[ "$SKIP_VALUE_INFER" -eq 0 ]]; then
  OUTPUT_PATHS_TO_CHECK+=("$INFER_OUTPUT_DIR")
fi
if [[ "$SKIP_INSTANCE_TABLE_BUILD" -eq 0 ]]; then
  OUTPUT_PATHS_TO_CHECK+=("$ACP_INSTANCE_TABLE_PATH")
fi
if [[ "$SKIP_POLICY_TRAIN" -eq 0 ]]; then
  if [[ "$POLICY_RESUME" -eq 0 ]]; then
    OUTPUT_PATHS_TO_CHECK+=("$POLICY_OUTPUT_DIR")
  fi
fi

for path in "${OUTPUT_PATHS_TO_CHECK[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Output path already exists, refusing to overwrite: $path" >&2
    exit 1
  fi
done

echo "Pipeline configuration:"
echo "  dataset_root:        $DATASET_ROOT"
echo "  dataset_repo_id:     $DATASET_REPO_ID"
echo "  work_dataset_root:   $WORK_DATASET_ROOT"
echo "  run_tag:             $RUN_TAG"
echo "  field_tag:           $FIELD_TAG"
echo "  value_field:         $VALUE_FIELD"
echo "  advantage_field:     $ADVANTAGE_FIELD"
echo "  indicator_field:     $INDICATOR_FIELD"
echo "  skip_value_train:    $SKIP_VALUE_TRAIN"
echo "  skip_value_infer:    $SKIP_VALUE_INFER"
echo "  skip_instance_table: $SKIP_INSTANCE_TABLE_BUILD"
echo "  skip_policy_train:   $SKIP_POLICY_TRAIN"
echo "  policy_resume:       $POLICY_RESUME"
echo "  value_train_gpu:     $VALUE_TRAIN_GPU"
echo "  infer_gpus:          $INFER_GPUS"
echo "  value_output_dir:    $VALUE_OUTPUT_DIR"
echo "  value_checkpoint:    $VALUE_CHECKPOINT_PATH"
echo "  value_checkpoint_ref:$VALUE_CHECKPOINT_REF"
echo "  infer_output_dir:    $INFER_OUTPUT_DIR"
echo "  acp_instance_table:  $ACP_INSTANCE_TABLE_PATH"
echo "  negative_bottom:     $ACP_NEGATIVE_BOTTOM_RATIO"
echo "  positive_top_dup:    $ACP_POSITIVE_TOP_DUPLICATE_RATIO"
echo "  policy_output_dir:   $POLICY_OUTPUT_DIR"
echo "  policy_config_path:  ${POLICY_CONFIG_PATH:-<none>}"
echo "  c_fail_coef:         $C_FAIL_COEF"
echo "  length_scale_quantile:$LENGTH_SCALE_QUANTILE"

if [[ "$SKIP_VALUE_TRAIN" -eq 0 ]]; then
  VALUE_TRAIN_ARGS=(
    -m lerobot.scripts.lerobot_value_train
    --value.type="$VALUE_TYPE"
    --value.backbone_source=pi05
    --value.pi05_repo_id=lerobot/pi05_base
    --value.dtype="$VALUE_DTYPE"
    --value.push_to_hub=false
    --batch_size="$VALUE_BATCH_SIZE"
    --steps="$VALUE_STEPS"
    --save_freq="$VALUE_SAVE_FREQ"
    --save_checkpoint=true
    --wandb.enable="$WANDB_ENABLE"
    --wandb.disable_artifact=true
    --job_name="$VALUE_JOB_NAME"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$WORK_DATASET_ROOT"
    --output_dir="$VALUE_OUTPUT_DIR"
    --targets.c_fail_coef="$C_FAIL_COEF"
    --targets.length_scale_quantile="$LENGTH_SCALE_QUANTILE"
    --value.freeze_language_model="$FREEZE_LANGUAGE_MODEL"
    --value.freeze_vision_encoder="$FREEZE_VISION_ENCODER"
  )

  run_cmd env PYTHONPATH="$REPO_PYTHONPATH" CUDA_VISIBLE_DEVICES="$VALUE_TRAIN_GPU" accelerate launch \
    --mixed_precision="$MIXED_PRECISION" \
    "${VALUE_TRAIN_ARGS[@]}"
else
  echo "Skipping value training."
fi

if [[ "$SKIP_VALUE_INFER" -eq 0 ]]; then
  if [[ "$INFER_GPUS" == "all" ]]; then
    unset CUDA_VISIBLE_DEVICES
    VALUE_INFER_NUM_PROCESSES="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$VALUE_INFER_NUM_PROCESSES" -lt 1 ]]; then
      echo "No CUDA devices detected for value inference." >&2
      exit 1
    fi
    VALUE_INFER_ENV_CMD=(env "PYTHONPATH=$REPO_PYTHONPATH")
  else
    VALUE_INFER_NUM_PROCESSES="$(awk -F',' '{print NF}' <<<"$INFER_GPUS")"
    VALUE_INFER_ENV_CMD=(env "PYTHONPATH=$REPO_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$INFER_GPUS")
  fi

  VALUE_INFER_ARGS=(
    -m lerobot.scripts.lerobot_value_infer
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$WORK_DATASET_ROOT"
    --dataset.download_videos=false
    --inference.checkpoint_path="$VALUE_CHECKPOINT_PATH"
    --inference.checkpoint_ref="$VALUE_CHECKPOINT_REF"
    --runtime.device=cuda
    --runtime.batch_size="$INFER_BATCH_SIZE"
    --acp.enable=true
    --acp.n_step="$ACP_N_STEP"
    --acp.positive_ratio="$ACP_POSITIVE_RATIO"
    --acp.c_fail_coef="$C_FAIL_COEF"
    --acp.length_scale_quantile="$LENGTH_SCALE_QUANTILE"
    --acp.value_field="$VALUE_FIELD"
    --acp.advantage_field="$ADVANTAGE_FIELD"
    --acp.indicator_field="$INDICATOR_FIELD"
    --output_dir="$INFER_OUTPUT_DIR"
    --job_name="$INFER_JOB_NAME"
  )

  if [[ "$VALUE_INFER_NUM_PROCESSES" -gt 1 ]]; then
    run_cmd "${VALUE_INFER_ENV_CMD[@]}" accelerate launch \
      --multi_gpu \
      --num_processes="$VALUE_INFER_NUM_PROCESSES" \
      "${VALUE_INFER_ARGS[@]}"
  else
    run_cmd "${VALUE_INFER_ENV_CMD[@]}" python \
      "${VALUE_INFER_ARGS[@]}"
  fi
else
  echo "Skipping value inference / ACP annotation writing."
fi

if [[ "$SKIP_INSTANCE_TABLE_BUILD" -eq 0 ]]; then
  ACP_INSTANCE_ARGS=(
    -m lerobot.scripts.lerobot_build_acp_instance_table
    --dataset-root="$WORK_DATASET_ROOT"
    --repo-id="$DATASET_REPO_ID"
    --advantage-field="$ADVANTAGE_FIELD"
    --negative-bottom-ratio="$ACP_NEGATIVE_BOTTOM_RATIO"
    --positive-top-duplicate-ratio="$ACP_POSITIVE_TOP_DUPLICATE_RATIO"
    --output-path="$ACP_INSTANCE_TABLE_PATH"
    --overwrite
  )

  run_cmd env PYTHONPATH="$REPO_PYTHONPATH" python \
    "${ACP_INSTANCE_ARGS[@]}"
else
  echo "Skipping ACP instance table build."
fi

if [[ "$SKIP_POLICY_TRAIN" -eq 0 ]]; then
  if [[ "$POLICY_GPUS" == "all" ]]; then
    unset CUDA_VISIBLE_DEVICES
    POLICY_NUM_PROCESSES="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "$POLICY_NUM_PROCESSES" -lt 1 ]]; then
      echo "No CUDA devices detected for policy training." >&2
      exit 1
    fi
    POLICY_ENV_CMD=(env "PYTHONPATH=$REPO_PYTHONPATH")
  else
    POLICY_NUM_PROCESSES="$(awk -F',' '{print NF}' <<<"$POLICY_GPUS")"
    POLICY_ENV_CMD=(env "PYTHONPATH=$REPO_PYTHONPATH" "CUDA_VISIBLE_DEVICES=$POLICY_GPUS")
  fi

  POLICY_TRAIN_ARGS=(
    --mixed_precision="$MIXED_PRECISION"
    -m lerobot.scripts.lerobot_train
    --dataset.use_imagenet_stats=false
    --policy.device=cuda
    --wandb.enable="$WANDB_ENABLE"
    --wandb.disable_artifact=true
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$WORK_DATASET_ROOT"
    --acp.enable=true
    --acp.indicator_field="$INDICATOR_FIELD"
    --acp.instance_table_path="$ACP_INSTANCE_TABLE_PATH"
    --acp.indicator_dropout_prob="$INDICATOR_DROPOUT_PROB"
  )

  if [[ "$POLICY_RESUME" -eq 1 ]]; then
    POLICY_TRAIN_ARGS+=(
      --resume=true
      --config_path="$POLICY_CONFIG_PATH"
    )
  else
    POLICY_TRAIN_ARGS+=(
      --policy.type="$POLICY_TYPE"
      --policy.pretrained_path="$POLICY_PRETRAINED_PATH"
      --steps="$POLICY_STEPS"
      --batch_size="$POLICY_BATCH_SIZE"
      --policy.dtype="$POLICY_DTYPE"
      --policy.gradient_checkpointing=true
      --policy.push_to_hub=false
      --save_freq="$POLICY_SAVE_FREQ"
      --job_name="$POLICY_JOB_NAME"
      --output_dir="$POLICY_OUTPUT_DIR"
    )
  fi

  if [[ "$POLICY_NUM_PROCESSES" -gt 1 ]]; then
    run_cmd "${POLICY_ENV_CMD[@]}" accelerate launch \
      --multi_gpu \
      --num_processes="$POLICY_NUM_PROCESSES" \
      "${POLICY_TRAIN_ARGS[@]}"
  else
    run_cmd "${POLICY_ENV_CMD[@]}" accelerate launch \
      "${POLICY_TRAIN_ARGS[@]}"
  fi
else
  echo "Skipping policy training."
fi

echo
echo "Pipeline completed."
echo "  backup/or working dataset: $WORK_DATASET_ROOT"
echo "  value train output:        $VALUE_OUTPUT_DIR"
echo "  value infer output:        $INFER_OUTPUT_DIR"
echo "  acp instance table:        $ACP_INSTANCE_TABLE_PATH"
echo "  policy train output:       $POLICY_OUTPUT_DIR"
