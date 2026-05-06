#!/usr/bin/env bash
# Dual-arm Pi0.5 + asynchronous RTC 推理启动脚本（参数布局参照 scripts/inference_socks.sh）。
# RTC 相关选项对应 src/lerobot/scripts/lerobot_arx5_dual_infer_rtc.py：
#   --rtc-s-min                  自上次 merge 后至少消费多少步才允许下一轮推理（默认 10）
#   --rtc-inference-delay-min    d_used 下界：max(本参数, delay 历史 max)（默认 1）
#   --rtc-delay-history-maxlen   索引差 delay 的滑动窗口长度（默认 64）
#   --rtc-max-guidance-weight    覆盖 RTC 引导强度上限；不设则脚本内用 10.0
#   --rtc-debug                  打开 RTCProcessor 调试轨迹

set -euo pipefail

python -u -m lerobot.scripts.lerobot_arx5_dual_infer_rtc \
  --task "Fold the socks together and place them on the edge of the desk" \
  --policy-path checkpoints/socks_short1400_v3/10k \
  --left-can-port can0 \
  --right-can-port can1 \
  --cameras base:254322073516 left_wrist:352122273239 right_wrist:335122271555 \
  --cam-width 424 --cam-height 240 \
  --duration 0.05 \
  --rtc-s-min 10 \
  --rtc-inference-delay-min 1 \
  --rtc-delay-history-maxlen 64 \
  --raw-train-record-dir /home/user/workspace/datasets/RL/raw_socks1400_v3_auto_test \
  2>&1 | grep -v "ARX方舟无限"

# 可选覆盖（与 RTC 相关，按需取消注释并接到上一命令）：
#   --rtc-max-guidance-weight 10.0
#   --rtc-debug

## 旧版同步 infer 对照（非 RTC）见 scripts/inference_socks.sh
## RTC 版不使用 --execution-horizon；安全段长由 checkpoint 的 n_action_steps 与安全模式 [I] 控制。
