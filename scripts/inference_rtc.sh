#!/usr/bin/env bash

python -u -m lerobot.scripts.lerobot_arx5_dual_infer_rtc \
--task "Use the handle to collect coffee grounds, twist it onto the machine, position the cup to brew the coffee, then move to desk and pour in the milk" \
--policy-path checkpoints/coffee_0510_10000/ \
--left-can-port can0 \
--right-can-port can1 \
--cameras base:254322070053 left_wrist:409122272986 right_wrist:352122274400 \
--cam-width 424 --cam-height 240 \
--duration 0.05 \
--execution-horizon 30 \
--rtc-inference-delay-steps 4 \
--rtc-prefetch-threshold 8 \
--rtc-auto-delay \
--rtc-auto-delay-window 3 \
--rtc-prefix-attention-schedule ONES \
--rtc-max-guidance-weight 10.0 \
--debug-mode \
--debug-pause-interval 15 \
--debug-log-dir logs/rtc_debug \
2>&1 | grep -v "ARX方舟无限"
