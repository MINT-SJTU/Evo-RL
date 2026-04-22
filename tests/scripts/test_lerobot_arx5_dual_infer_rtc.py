from lerobot.scripts.lerobot_arx5_dual_infer_rtc import _serialize_dual_arm_targets


def test_rtc_serialize_dual_arm_targets_matches_dataset_order():
    q_cmd_by_arm = {
        "left_arm": [101, 102, 103, 104, 105, 106],
        "right_arm": [1, 2, 3, 4, 5, 6],
    }
    gripper_by_arm = {
        "left_arm": 107,
        "right_arm": 7,
    }

    assert _serialize_dual_arm_targets(q_cmd_by_arm, gripper_by_arm) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
    ]
