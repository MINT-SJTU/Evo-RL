# Evo1 Known Issues

## 尚未完善的部分

- 多 embodiment 训练目前还没有完整打通。
  - 模型侧保留了 `embodiment_id`、`num_categories`、`default_embodiment_id` 这些接口。
  - 但当前常用数据集没有稳定的 `embodiment_id` 字段，训练时实际上仍然会回落到单一 `default_embodiment_id=0`。
  - 如果后续要做多机器人/多 embodiment 联训，还需要同时补齐：
    - 数据集里的稳定 `embodiment_id` 来源
    - 多 embodiment 的归一化统计，而不是单份聚合 stats
    - training config 层面对多数据源联训的明确支持

- 目前接入主要面向 Evo-RL / LeRobot 风格的 checkpoint。
  - 也就是带有 `config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json` 的目录。
  - 原始 Evo1 / DeepSpeed checkpoint 没有直接作为 Evo-RL 训练续训格式来兼容。

- 当前验证过、最稳定的训练路径是 4 卡 bf16。
  - 目录里只保留了四卡 accelerate 配置。
  - 单卡、双卡虽然理论上可以再配，但当前没有作为维护目标继续保留。

- 图像增强路径当前不作为默认推荐配置。
  - 已验证训练命令里默认使用 `dataset.image_transforms.enable=false`。
  - 原因不是 Evo1 算法本身依赖关闭增强，而是当前实验环境里的 transforms 路径没有作为稳定组合继续维护。

## 相比原始 Evo1 的工程差异

- 训练入口不再走原始 Evo1 的“大训练脚本”。
  - 现在是把 Evo1 接到了 LeRobot 统一训练框架里，通过 `lerobot_train` 启动。
  - `stage1/stage2`、checkpoint 保存、processor、dataset 读取、日志和训练配置都尽量复用 LeRobot 现有基础设施。

- `stage1 -> stage2` 的衔接方式改成了 Evo-RL 里的 `resume_pretrain=true` 语义。
  - 含义是：只加载权重，不恢复 optimizer / scheduler / global step。
  - 这和原始 Evo1 的“两阶段训练切换”意图是一致的，但实现方式更贴近 LeRobot 的配置体系。

- 混合精度执行边界比原始 Evo1 更统一。
  - 原始 Evo1 里，VLM embedding 和后面的 action head 前向不在同一个统一的训练 AMP 作用域里。
  - 现在 Evo-RL 版是在更大的 `accelerator.autocast()` 范围中执行整个 policy forward。
  - 这属于工程上的规整化，主要影响的是执行路径、速度和数值边界；在 `stage2` finetune VLM 时尤其值得注意。

- 分布式训练实现不同。
  - 原始 Evo1 训练主要依赖 DeepSpeed。
  - 现在的 Evo-RL 接入版默认走 Accelerate bf16 配置。
  - 这不改变 Evo1 的核心模型结构和两阶段训练思路，但会让工程实现、混精边界、分布式包装方式与原始仓库有所不同。

- processor / preprocessor / postprocessor 被纳入了 LeRobot 的标准保存格式。
  - 这样推理、录制和训练可以复用同一套加载接口。
  - 也意味着 Evo1 不再只是一套训练脚本，而是完整地成为 LeRobot policy registry 里的一个 policy。

- 数据读取不再依赖原始 Evo1 自己的大型 dataset 训练脚本。
  - 当前训练直接复用 LeRobot 数据集接口。
  - 为了对照复现原始 Evo1 时，额外写过一个 v3.0 数据到原始 Evo1 loader 可读格式的适配脚本；但这条适配链不是 Evo-RL 日常训练的默认路径。
