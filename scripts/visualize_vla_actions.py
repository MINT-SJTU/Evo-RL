#!/usr/bin/env python3
"""Visualize VLA action predictions vs ground truth for a dataset episode.

Loads a LeRobot dataset episode and runs pi0.5 (and optionally RLT) inference
on each frame, then generates an interactive HTML page showing action curves
in a 30-frame sliding window with a scrub bar.

Usage:
    PYTHONPATH=src python scripts/visualize_vla_actions.py \
        --dataset-path /path/to/dataset \
        --vla-model /path/to/pi05_sft \
        --stride 1 --output action_viz.html

    # With RLT:
    PYTHONPATH=src python scripts/visualize_vla_actions.py \
        --dataset-path /path/to/dataset \
        --vla-model /path/to/pi05_sft \
        --rl-token-ckpt checkpoints/rlt/demo_adapt_checkpoint.pt \
        --ac-ckpt checkpoints/rlt/rl_checkpoint.pt \
        --actor-hidden-dim 512 --actor-num-layers 4 \
        --stride 1 --output action_viz.html
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import copy
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

log = logging.getLogger(__name__)

JOINT_TYPES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
DEFAULT_JOINT_NAMES = (
    [f"left_{j}.pos" for j in JOINT_TYPES] + [f"right_{j}.pos" for j in JOINT_TYPES]
)
DEFAULT_CAMERA_KEYS = ["left_wrist", "right_wrist", "right_front"]
WINDOW = 30  # frames per visible window


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize VLA action predictions")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--vla-model", type=str, required=True)
    p.add_argument("--rl-token-ckpt", type=str, default="")
    p.add_argument("--ac-ckpt", type=str, default="")
    p.add_argument("--task", type=str, default="Insert the copper screw into the black sleeve.")
    p.add_argument("--output", type=str, default="action_viz.html")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--repo-id", type=str, default="viz_dataset")
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--actor-hidden-dim", type=int, default=256)
    p.add_argument("--actor-num-layers", type=int, default=3)
    p.add_argument("--actor-residual", action="store_true", default=True)
    p.add_argument("--actor-activation", type=str, default="relu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: str, repo_id: str, episode_index: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id, root=dataset_path,
        episodes=[episode_index], video_backend="pyav",
    )


def detect_camera_keys(dataset) -> list[str]:
    prefix = "observation.images."
    return [k.removeprefix(prefix) for k in dataset.features if k.startswith(prefix)]


# ---------------------------------------------------------------------------
# Unnormalize: QUANTILES mode (q01, q99) -> original space
# ---------------------------------------------------------------------------


def load_action_quantiles(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Load q01 and q99 from dataset stats for action unnormalization."""
    stats = dataset.meta.stats["action"]
    q01 = stats["q01"].numpy() if isinstance(stats["q01"], torch.Tensor) else np.array(stats["q01"])
    q99 = stats["q99"].numpy() if isinstance(stats["q99"], torch.Tensor) else np.array(stats["q99"])
    return q01, q99


def unnormalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of QUANTILES normalization: [-1, 1] -> original range via q01/q99."""
    return (actions + 1.0) / 2.0 * (q99 - q01) + q01


# ---------------------------------------------------------------------------
# Observation conversion
# ---------------------------------------------------------------------------


def item_to_numpy_obs(item: dict, camera_full_keys: list[str]) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for full_key in camera_full_keys:
        img = item[full_key]
        obs[full_key] = (img.permute(1, 2, 0) * 255).byte().numpy()
    obs["observation.state"] = item["observation.state"].numpy()
    return obs


def item_to_tensor_obs(item: dict, camera_full_keys: list[str]) -> dict[str, torch.Tensor]:
    """Convert dataset item to tensor dict for RLT obs_bridge.

    Splits observation.state into per-joint keys (e.g. left_shoulder_pan.pos)
    because robot_obs_to_rlt_obs expects individual joint keys.
    """
    obs: dict[str, torch.Tensor] = {}
    for full_key in camera_full_keys:
        obs[full_key] = item[full_key]
    state = item["observation.state"]
    for i, jname in enumerate(DEFAULT_JOINT_NAMES):
        obs[jname] = state[i]
    return obs


# ---------------------------------------------------------------------------
# Pi0.5 inference
# ---------------------------------------------------------------------------


def load_pi05_policy(vla_model: str, dataset, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats

    cfg = PreTrainedConfig.from_pretrained(vla_model)
    cfg.pretrained_path = vla_model
    cfg.device = device
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=vla_model,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
    )
    return policy, preprocessor


def run_pi05_inference(
    policy, preprocessor, dataset, camera_full_keys: list[str],
    task: str, device: str, stride: int,
) -> np.ndarray:
    """Returns (N, action_dim) in NORMALIZED [-1,1] space."""
    from lerobot.policies.utils import prepare_observation_for_inference

    actions = []
    frame_indices = list(range(0, len(dataset), stride))
    log.info("pi0.5: %d frames (stride=%d)", len(frame_indices), stride)

    t0 = time.monotonic()
    for idx in tqdm(frame_indices, desc="pi0.5"):
        item = dataset[idx]
        obs_np = item_to_numpy_obs(item, camera_full_keys)
        obs_prepared = prepare_observation_for_inference(copy(obs_np), torch.device(device), task)
        obs_prepared = preprocessor(obs_prepared)
        policy._action_queue.clear()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            chunk = policy.predict_action_chunk(obs_prepared)
        actions.append(chunk[0, 0, :].cpu().numpy())

    elapsed = time.monotonic() - t0
    log.info("pi0.5 done: %.1fs (%.0f ms/frame)", elapsed, elapsed / max(len(actions), 1) * 1000)
    return np.array(actions)


# ---------------------------------------------------------------------------
# RLT inference
# ---------------------------------------------------------------------------


def load_rlt_policy(args: argparse.Namespace, camera_keys: list[str]):
    from lerobot.rlt.deploy import load_rlt_deploy_policy
    from lerobot.rlt.deploy_config import DeployConfig

    deploy_cfg = DeployConfig(
        vla_model_path=args.vla_model,
        rl_token_checkpoint=args.rl_token_ckpt,
        ac_checkpoint=args.ac_ckpt,
        camera_keys=camera_keys,
        proprio_keys=DEFAULT_JOINT_NAMES,
        action_keys=DEFAULT_JOINT_NAMES,
        phase_mode="always_rl",
        deterministic=True,
        device=args.device,
        task_instruction=args.task,
        token_pool_size=64, chunk_length=10,
        tokenizer_path=args.tokenizer_path,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        actor_residual=args.actor_residual,
        actor_activation=args.actor_activation,
    )
    return load_rlt_deploy_policy(deploy_cfg)


def run_rlt_inference(
    rlt_policy, dataset, camera_keys: list[str], camera_full_keys: list[str],
    device: str, stride: int,
) -> np.ndarray:
    """Returns (N, action_dim) in NORMALIZED [-1,1] space."""
    from lerobot.rlt.obs_bridge import robot_obs_to_rlt_obs

    actions = []
    frame_indices = list(range(0, len(dataset), stride))
    log.info("RLT: %d frames (stride=%d)", len(frame_indices), stride)

    t0 = time.monotonic()
    for idx in tqdm(frame_indices, desc="RLT"):
        item = dataset[idx]
        obs_dict = item_to_tensor_obs(item, camera_full_keys)
        obs_rlt = robot_obs_to_rlt_obs(obs_dict, camera_keys, DEFAULT_JOINT_NAMES, device)
        with torch.inference_mode():
            chunk, _, _ = rlt_policy._compute_action_chunk(obs_rlt)
        actions.append(chunk[0, 0, :].cpu().numpy())

    elapsed = time.monotonic() - t0
    log.info("RLT done: %.1fs (%.0f ms/frame)", elapsed, elapsed / max(len(actions), 1) * 1000)
    return np.array(actions)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def extract_ground_truth(dataset, stride: int) -> np.ndarray:
    return np.array([dataset[i]["action"].numpy() for i in range(0, len(dataset), stride)])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

COLORS = {"GT": "#1f77b4", "pi0.5": "#ff7f0e", "RLT": "#2ca02c"}


def build_stats_html(gt: np.ndarray, pi05: np.ndarray, rlt: np.ndarray | None) -> str:
    dim = gt.shape[1]
    names = DEFAULT_JOINT_NAMES[:dim]
    has_rlt = rlt is not None
    f = lambda v: f"{v:.2f}"  # noqa: E731
    td = '<td style="padding:2px 6px;border:1px solid #ddd;text-align:right;">'

    hdr = "<tr><th>Joint</th><th>GT μ</th><th>GT σ</th><th>pi0.5 μ</th><th>pi0.5 σ</th><th>MAE(pi0.5)</th>"
    if has_rlt:
        hdr += "<th>RLT μ</th><th>RLT σ</th><th>MAE(RLT)</th>"
    hdr += "</tr>"

    rows = []
    for j, name in enumerate(names):
        g, p = gt[:, j], pi05[:, j]
        row = (
            f'{td}{name}</td>'
            f'{td}{f(g.mean())}</td>{td}{f(g.std())}</td>'
            f'{td}{f(p.mean())}</td>{td}{f(p.std())}</td>'
            f'{td}<b>{f(np.mean(np.abs(p - g)))}</b></td>'
        )
        if has_rlt:
            r = rlt[:, j]
            row += (
                f'{td}{f(r.mean())}</td>{td}{f(r.std())}</td>'
                f'{td}<b>{f(np.mean(np.abs(r - g)))}</b></td>'
            )
        rows.append(f"<tr>{row}</tr>")

    return (
        '<h3 style="font-family:sans-serif;">Per-Joint Statistics (degrees)</h3>'
        '<table style="border-collapse:collapse;font-family:monospace;font-size:13px;">'
        f'<thead style="background:#f0f0f0;">{hdr}</thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


# ---------------------------------------------------------------------------
# HTML generation: 30-frame sliding window with scrub bar
# ---------------------------------------------------------------------------


def build_html(
    gt: np.ndarray, pi05: np.ndarray, rlt: np.ndarray | None,
    stride: int, fps: float, stats_html: str,
) -> str:
    """Generate self-contained HTML with plotly + vanilla JS slider for 30-frame window."""
    n = gt.shape[0]
    time_s = (np.arange(n) * stride / fps).tolist()

    # Prepare data as JSON for the JS side
    sources = {"GT": gt.tolist(), "pi05": pi05.tolist()}
    if rlt is not None:
        sources["RLT"] = rlt.tolist()

    data_json = json.dumps({
        "time": time_s,
        "sources": sources,
        "n": n,
        "stride": stride,
        "fps": fps,
        "joint_types": JOINT_TYPES,
        "window": WINDOW,
        "colors": COLORS,
    })

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VLA Action Visualization</title>
<script src="https://cdn.plot.ly/plotly-3.5.0.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; }}
  #slider-container {{ padding: 10px 40px; }}
  #frame-slider {{ width: 100%; cursor: pointer; }}
  #frame-info {{ text-align: center; margin: 5px 0; font-size: 14px; }}
  .stats-container {{ padding: 20px; }}
</style>
</head><body>
<div id="plot"></div>
<div id="slider-container">
  <div id="frame-info">Frame 0 - {WINDOW-1} / {n-1} &nbsp; (0.00s - {(WINDOW-1)*stride/fps:.2f}s)</div>
  <input type="range" id="frame-slider" min="0" max="{max(0, n - WINDOW)}" value="0" step="1">
</div>
<div class="stats-container">{stats_html}</div>

<script>
const D = {data_json};
const W = D.window;
const JT = D.joint_types;
const SRC_KEYS = Object.keys(D.sources);
const DASH = {{GT: "solid", pi05: "solid", RLT: "solid"}};
const CLR = {{GT: D.colors.GT, pi05: D.colors["pi0.5"], RLT: D.colors.RLT}};
const LABEL = {{GT: "GT", pi05: "pi0.5", RLT: "RLT"}};

function buildTraces(start) {{
  const end = Math.min(start + W, D.n);
  const t = D.time.slice(start, end);
  const traces = [];
  for (let ji = 0; ji < 6; ji++) {{
    const li = ji, ri = ji + 6;
    const row = Math.floor(ji / 2) + 1;
    const col = (ji % 2) + 1;
    for (const sk of SRC_KEYS) {{
      const arr = D.sources[sk];
      const yL = arr.slice(start, end).map(r => r[li]);
      const yR = arr.slice(start, end).map(r => r[ri]);
      traces.push({{
        x: t, y: yL, type: "scatter", mode: "lines",
        name: LABEL[sk] + " L", legendgroup: LABEL[sk] + "_L",
        line: {{color: CLR[sk], dash: "solid", width: 1.5}},
        showlegend: ji === 0,
        xaxis: "x" + (ji > 0 ? (ji+1) : ""),
        yaxis: "y" + (ji > 0 ? (ji+1) : ""),
        hovertemplate: LABEL[sk] + " L_" + JT[ji] + ": %{{y:.2f}}<extra></extra>",
      }});
      traces.push({{
        x: t, y: yR, type: "scatter", mode: "lines",
        name: LABEL[sk] + " R", legendgroup: LABEL[sk] + "_R",
        line: {{color: CLR[sk], dash: "dash", width: 1.5}},
        showlegend: ji === 0,
        xaxis: "x" + (ji > 0 ? (ji+1) : ""),
        yaxis: "y" + (ji > 0 ? (ji+1) : ""),
        hovertemplate: LABEL[sk] + " R_" + JT[ji] + ": %{{y:.2f}}<extra></extra>",
      }});
    }}
  }}
  return traces;
}}

function makeLayout() {{
  const layout = {{
    height: 900, width: 1200,
    template: "plotly_white",
    grid: {{rows: 3, columns: 2, pattern: "independent", roworder: "top to bottom"}},
    legend: {{orientation: "h", y: -0.05, x: 0.5, xanchor: "center"}},
    margin: {{t: 40, b: 60}},
  }};
  for (let ji = 0; ji < 6; ji++) {{
    const ax = ji > 0 ? (ji+1) : "";
    layout["xaxis" + ax] = {{title: "time (s)"}};
    layout["yaxis" + ax] = {{title: JT[ji].replace(/_/g, " ")}};
  }}
  // subplot titles via annotations
  for (let ji = 0; ji < 6; ji++) {{
    if (!layout.annotations) layout.annotations = [];
    const col = ji % 2, row = Math.floor(ji / 2);
    layout.annotations.push({{
      text: JT[ji].replace(/_/g, " ").replace(/\\b\\w/g, c => c.toUpperCase()),
      xref: "paper", yref: "paper",
      x: col * 0.55 + 0.22, y: 1.0 - row * 0.34,
      showarrow: false, font: {{size: 13}},
    }});
  }}
  return layout;
}}

const plotDiv = document.getElementById("plot");
Plotly.newPlot(plotDiv, buildTraces(0), makeLayout(), {{responsive: true}});

const slider = document.getElementById("frame-slider");
const info = document.getElementById("frame-info");
slider.addEventListener("input", function() {{
  const start = parseInt(this.value);
  const end = Math.min(start + W, D.n) - 1;
  const tStart = D.time[start].toFixed(2);
  const tEnd = D.time[end].toFixed(2);
  info.textContent = "Frame " + start + " - " + end + " / " + (D.n-1) + "  (" + tStart + "s - " + tEnd + "s)";
  Plotly.react(plotDiv, buildTraces(start), makeLayout());
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    has_rlt = bool(args.rl_token_ckpt and args.ac_ckpt)

    # 1. Dataset
    dataset = load_dataset(args.dataset_path, args.repo_id, args.episode_index)
    camera_keys = detect_camera_keys(dataset)
    camera_full_keys = [f"observation.images.{k}" for k in camera_keys]
    fps = dataset.meta.fps
    log.info("Cameras: %s, fps=%d, frames=%d", camera_keys, fps, len(dataset))

    # Load unnormalization stats
    q01, q99 = load_action_quantiles(dataset)
    log.info("Action q01[:4]=%s  q99[:4]=%s", q01[:4].tolist(), q99[:4].tolist())

    # 2. Ground truth (already in original space)
    gt_actions = extract_ground_truth(dataset, args.stride)
    log.info("GT shape: %s, range=[%.1f, %.1f]", gt_actions.shape, gt_actions.min(), gt_actions.max())

    # 3. Pi0.5 inference (normalized) -> unnormalize
    log.info("Loading pi0.5 from %s ...", args.vla_model)
    pi05_policy, preprocessor = load_pi05_policy(args.vla_model, dataset, args.device)
    pi05_norm = run_pi05_inference(
        pi05_policy, preprocessor, dataset, camera_full_keys,
        args.task, args.device, args.stride,
    )
    pi05_actions = unnormalize_actions(pi05_norm, q01, q99)
    log.info("pi0.5 unnormalized range: [%.1f, %.1f]", pi05_actions.min(), pi05_actions.max())
    del pi05_policy, preprocessor
    torch.cuda.empty_cache()
    import gc; gc.collect()  # free CPU RAM before loading RLT

    # 4. RLT inference (normalized) -> unnormalize
    rlt_actions = None
    if has_rlt:
        log.info("Loading RLT rl_token=%s ac=%s ...", args.rl_token_ckpt, args.ac_ckpt)
        rlt_policy = load_rlt_policy(args, camera_keys)
        rlt_norm = run_rlt_inference(
            rlt_policy, dataset, camera_keys, camera_full_keys, args.device, args.stride,
        )
        rlt_actions = unnormalize_actions(rlt_norm, q01, q99)
        log.info("RLT unnormalized range: [%.1f, %.1f]", rlt_actions.min(), rlt_actions.max())
        del rlt_policy
        torch.cuda.empty_cache()

    # 5. Truncate to common action_dim
    dim = min(gt_actions.shape[1], pi05_actions.shape[1])
    if rlt_actions is not None:
        dim = min(dim, rlt_actions.shape[1])
    gt_actions, pi05_actions = gt_actions[:, :dim], pi05_actions[:, :dim]
    if rlt_actions is not None:
        rlt_actions = rlt_actions[:, :dim]

    # 6. Generate HTML
    stats = build_stats_html(gt_actions, pi05_actions, rlt_actions)
    html = build_html(gt_actions, pi05_actions, rlt_actions, args.stride, fps, stats)
    Path(args.output).write_text(html, encoding="utf-8")
    log.info("Written to %s", args.output)


if __name__ == "__main__":
    main()
