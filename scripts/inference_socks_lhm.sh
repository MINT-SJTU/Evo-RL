# python -u -m lerobot.scripts.lerobot_arx5_dual_infer   \
# --task "Fold the towel and put it on the edge of the table"   \
# --policy-path checkpoints/towel300_0408/pretrained_model \
# --left-can-port can0   \
# --right-can-port can1  \
# --cameras base:150622073629 left_wrist:352122273179  right_wrist:409122272986 \
# --cam-width 424 --cam-height 240 \
# --duration 0.05 \
# --raw-train-record-dir /home/user/workspace/zsj/datasets/arx5/raw_socks_robot \
# --joint-traj-dir traj/joint_traj/
# 2>&1 | grep -v "ARX方舟无限"
# --end-traj-dir /home/user/workspace/zsj/Evo-RL/traj
# --execution-horizon 10 --RTC 

python -u -m lerobot.scripts.lerobot_arx5_dual_infer   \
--task "Fold the socks together and place them on the edge of the desk"   \
--policy-path checkpoints/socks600_short_v3/16k/ \
--left-can-port can0 \
--right-can-port can1 \
--cameras base:254522076820 left_wrist:352122273179  right_wrist:352122270765 \
--cam-width 424 --cam-height 240 \
--duration 0.05 \
--raw-train-record-dir /home/user/workspace/zsj/datasets/arx5/socks600_short_v3_10k \
--execution-horizon 25 \
2>&1 | grep -v "ARX方舟无限"
## --joint-traj-dir traj/joint_traj/ 
