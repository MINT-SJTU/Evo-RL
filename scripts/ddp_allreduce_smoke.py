#!/usr/bin/env python3
"""Small distributed all-reduce smoke test for NCCL/Gloo setup."""

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--mb", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    host = socket.gethostname()

    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    t0 = time.time()
    dist.init_process_group(backend=args.backend)
    init_s = time.time() - t0

    numel = max(1, args.mb * 1024 * 1024 // 4)
    x = torch.full((numel,), rank + 1, dtype=torch.float32, device=device)

    if rank == 0:
        print(
            f"[SMOKE] backend={args.backend} world={world_size} "
            f"iters={args.iters} tensor_mb={args.mb}",
            flush=True,
        )

    for i in range(args.iters):
        if args.backend == "nccl":
            torch.cuda.synchronize()
        t1 = time.time()
        dist.all_reduce(x)
        if args.backend == "nccl":
            torch.cuda.synchronize()
        elapsed = time.time() - t1
        expected = world_size * (world_size + 1) / 2
        got = float(x[0].item())
        ok = abs(got - expected) < 1e-3
        print(
            f"[SMOKE] host={host} rank={rank:02d} local={local_rank} "
            f"init_s={init_s:.2f} iter={i} allreduce_s={elapsed:.4f} "
            f"value={got:.1f} ok={ok}",
            flush=True,
        )
        x.fill_(rank + 1)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
