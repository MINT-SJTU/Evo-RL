#!/usr/bin/env bash
set -euo pipefail
interfaces=(can0 can1 can2 can3)  # can0=left follower, can1=right follower, can2=left leader, can3=right leader
for iface in "${interfaces[@]}"; do
  echo "Configuring $iface..."
  sudo ip link set "$iface" down 2>/dev/null || true
  if sudo ip link set "$iface" type can bitrate 1000000 dbitrate 5000000 fd on 2>/dev/null; then
    echo "$iface: CAN FD"
  else
    echo "$iface: CAN FD unsupported, using CAN 2.0 at 1 Mbps"
    sudo ip link set "$iface" type can bitrate 1000000
  fi
  sudo ip link set "$iface" up
done
ip -br link show type can
