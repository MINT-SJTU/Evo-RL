python -u -m lerobot.scripts.lerobot_arx5_dual_infer_rtc \
--task "Fold the socks together and place them on the edge of the desk" \
--policy-path checkpoints/socks600_short_v3/16k/ \
--left-can-port can0 \
--right-can-port can1 \
--cameras base:254322073516 left_wrist:409122272986 right_wrist:335122271555 \
--cam-width 424 --cam-height 240 \
--duration 0.05 \
--RTC \
--rtc-s-min 14 \
--rtc-execution-horizon 25 \
--rtc-latency-buffer-len 12 \
--rtc-latency-default-steps 6 \
--raw-train-record-dir /home/user/workspace/zsj/datasets/arx5/raw_socks_0411 \
2>&1 | grep -v "ARX方舟无限"
