import draccus
import pytest
from draccus.utils import ParsingError

from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig


def test_teleoperate_parses_bi_piperx_types():
    args = [
        "--robot.type=bi_piperx_follower",
        "--robot.id=my_bi_piperx_follower",
        "--robot.left_arm_config.port=can0",
        "--robot.right_arm_config.port=can2",
        "--teleop.type=bi_piperx_leader",
        "--teleop.id=my_bi_piperx_leader",
        "--teleop.left_arm_config.port=can1",
        "--teleop.right_arm_config.port=can3",
    ]
    cfg = draccus.parse(config_class=TeleoperateConfig, config_path=None, args=args)
    assert cfg.robot.type == "bi_piperx_follower"
    assert cfg.teleop.type == "bi_piperx_leader"


def test_record_rejects_mixed_bi_piper_pair():
    args = [
        "--robot.type=bi_piper_follower",
        "--robot.id=my_bi_piper_follower",
        "--robot.left_arm_config.port=can0",
        "--robot.right_arm_config.port=can2",
        "--teleop.type=bi_piperx_leader",
        "--teleop.id=my_bi_piperx_leader",
        "--teleop.left_arm_config.port=can1",
        "--teleop.right_arm_config.port=can3",
        "--dataset.repo_id=dummy/dummy",
        "--dataset.single_task=test",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=1",
        "--dataset.reset_time_s=1",
        "--dataset.push_to_hub=false",
    ]
    with pytest.raises(ParsingError) as exc_info:
        draccus.parse(config_class=RecordConfig, config_path=None, args=args)
    assert "Couldn't instantiate class RecordConfig" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "must be paired" in str(exc_info.value.__cause__)


def test_record_rejects_bimanual_piper_teleop_with_non_bimanual_robot():
    args = [
        "--robot.type=so101_follower",
        "--robot.port=/dev/mock",
        "--teleop.type=bi_piperx_leader",
        "--teleop.id=my_bi_piperx_leader",
        "--teleop.left_arm_config.port=can1",
        "--teleop.right_arm_config.port=can3",
        "--dataset.repo_id=dummy/dummy",
        "--dataset.single_task=test",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=1",
        "--dataset.reset_time_s=1",
        "--dataset.push_to_hub=false",
    ]
    with pytest.raises(ParsingError) as exc_info:
        draccus.parse(config_class=RecordConfig, config_path=None, args=args)
    assert "Couldn't instantiate class RecordConfig" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "must be paired" in str(exc_info.value.__cause__)
