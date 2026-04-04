from __future__ import annotations

import math

from lerobot.rlt.evaluator import evaluate_offline, OfflineEvalMetrics
from lerobot.rlt.trainer import offline_rl_loop
from tests.rlt.helpers import make_test_algorithm, fill_buffer


def test_evaluate_offline_returns_metrics():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    val_buf = fill_buffer(30)

    metrics = evaluate_offline(algorithm, val_buf, cfg, num_batches=3)

    assert isinstance(metrics, OfflineEvalMetrics)
    assert not math.isnan(metrics.expert_action_mse)
    assert not math.isnan(metrics.ref_action_mse)
    assert not math.isnan(metrics.mean_q_policy)
    assert not math.isnan(metrics.mean_q_expert)
    assert not math.isnan(metrics.q_gap)
    assert not math.isnan(metrics.mean_critic_td_error)
    assert metrics.expert_action_mse >= 0.0
    assert metrics.ref_action_mse >= 0.0


def test_evaluate_offline_q_gap_sign():
    """After some training, q_gap = Q(policy) - Q(expert) should not be NaN."""
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    buf = fill_buffer(50)

    offline_rl_loop(algorithm, cfg, buf)

    val_buf = fill_buffer(20)
    metrics = evaluate_offline(algorithm, val_buf, cfg, num_batches=5)

    assert not math.isnan(metrics.q_gap)
    # q_gap is mean_q_policy - mean_q_expert: should be a finite number
    assert math.isfinite(metrics.q_gap)
