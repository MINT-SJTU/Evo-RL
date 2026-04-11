python -u -m lerobot.scripts.lerobot_arx5_dual_infer   \
--task "Fold the towel and put it on the edge of the table"   \
--policy-path checkpoints/towel300_0408/pretrained_model \
--left-can-port can0   \
--right-can-port can1  \
--cameras base:150622073629 left_wrist:352122273179  right_wrist:409122272986 \
--cam-width 424 --cam-height 240 \
--RTC \
--duration 0.05 \
--raw-train-record-dir /home/user/workspace/zsj/datasets/arx5/raw_socks_robot \
--execution-horizon 10\
2>&1 | grep -v "ARX方舟无限"
# --end-traj-dir /home/user/workspace/zsj/Evo-RL/traj
