from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RLTRecordConfig, RecordConfig, record

logger = logging.getLogger(__name__)


@dataclass
class RLTDataCollectionConfig(RecordConfig):
    rlt: RLTRecordConfig = field(default_factory=lambda: RLTRecordConfig(enable=True))
    enable_episode_outcome_labeling: bool = True


@parser.wrap()
def rlt_record(cfg: RLTDataCollectionConfig) -> None:
    logger.info("Starting RLT data collection (rlt.enable=%s)", cfg.rlt.enable)
    record(cfg)


def main():
    rlt_record()


if __name__ == "__main__":
    main()
