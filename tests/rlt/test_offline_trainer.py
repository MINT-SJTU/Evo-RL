from __future__ import annotations

import math

import torch

from lerobot.rlt.trainer import offline_rl_loop
from tests.rlt.helpers import make_test_algorithm, fill_buffer


def test_offline_rl_loop_runs():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    buf = fill_buffer()

    metrics = offline_rl_loop(algorithm, cfg, buf)

    assert len(metrics.critic_losses) == cfg.offline_rl.num_gradient_steps
    assert len(metrics.actor_losses) > 0
    assert all(not math.isnan(l) for l in metrics.critic_losses)
    assert all(not math.isnan(l) for l in metrics.actor_losses)


def test_offline_rl_loop_frozen_params():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    vla_params_before = {n: p.data.clone() for n, p in algorithm.policy.vla.named_parameters()}
    enc_params_before = {n: p.data.clone() for n, p in algorithm.policy.rl_token.encoder.named_parameters()}
    rl_embed_before = algorithm.policy.rl_token.rl_token_embed.data.clone()

    buf = fill_buffer()
    offline_rl_loop(algorithm, cfg, buf)

    for name, p in algorithm.policy.vla.named_parameters():
        assert torch.equal(p.data, vla_params_before[name]), f"VLA param {name} changed"
    for name, p in algorithm.policy.rl_token.encoder.named_parameters():
        assert torch.equal(p.data, enc_params_before[name]), f"Encoder param {name} changed"
    assert torch.equal(algorithm.policy.rl_token.rl_token_embed.data, rl_embed_before)


def test_offline_rl_loop_actor_critic_update():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    actor_params_before = {n: p.data.clone() for n, p in algorithm.policy.actor.named_parameters()}
    critic_params_before = {n: p.data.clone() for n, p in algorithm.critic.named_parameters()}

    buf = fill_buffer()
    offline_rl_loop(algorithm, cfg, buf)

    actor_changed = any(
        not torch.equal(p.data, actor_params_before[n])
        for n, p in algorithm.policy.actor.named_parameters()
    )
    critic_changed = any(
        not torch.equal(p.data, critic_params_before[n])
        for n, p in algorithm.critic.named_parameters()
    )
    assert actor_changed, "Actor params did not change after training"
    assert critic_changed, "Critic params did not change after training"


def test_offline_rl_loop_with_val_buffer():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    train_buf = fill_buffer()
    val_buf = fill_buffer(20)

    metrics = offline_rl_loop(algorithm, cfg, train_buf, val_buffer=val_buf)

    assert len(metrics.critic_losses) == cfg.offline_rl.num_gradient_steps
