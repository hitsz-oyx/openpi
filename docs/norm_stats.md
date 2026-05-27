# 归一化统计信息

遵循常见实践，我们的模型在策略训练和推理过程中会对本体感受状态输入和动作目标进行归一化。用于归一化的统计信息是在训练数据上计算的，并与模型检查点一起存储。

## 重新加载归一化统计信息

当你在新的数据集上微调我们的模型时，你需要决定是 (A) 复用现有的归一化统计信息，还是 (B) 在新的训练数据上计算新的统计信息。哪种选择更适合你，取决于你的机器人和任务与预训练数据集中机器人和任务分布的相似程度。下面我们列出每个模型的所有可用预训练归一化统计信息。

**如果你的目标机器人与这些预训练统计信息之一匹配，请考虑重新加载相同的归一化统计信息。** 通过重新加载归一化统计信息，你数据集中的动作将更加"熟悉"模型，这可以带来更好的性能。你可以通过在训练配置中添加一个 `AssetsConfig` 来重新加载归一化统计信息，该配置指向相应的检查点目录和归一化统计信息 ID，以下是 `pi0_base` 检查点的 `Trossen`（又称 ALOHA）机器人统计信息的示例：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

有关重新加载归一化统计信息的完整训练配置示例，请参阅[训练配置文件](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)中的 `pi0_aloha_pen_uncap` 配置。

**注意：** 要成功重新加载归一化统计信息，你的机器人 + 数据集必须遵循预训练时使用的动作空间定义。我们在下面提供了动作空间定义的详细描述。

**注意 #2：** 重新加载归一化统计信息是否有益，取决于你的机器人和任务与预训练数据集中机器人和任务分布的相似程度。我们建议始终尝试两种方法：重新加载统计信息和使用在新数据集上计算的新统计信息进行训练（请参阅[主 README](../README.md) 获取如何计算新统计信息的说明），然后选择对你的任务效果更好的那个。


## 提供的预训练归一化统计信息

以下是所有预训练归一化统计信息的列表。我们为 `pi0_base` 和 `pi0_fast_base` 模型都提供了这些统计信息。对于 `pi0_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，请将 `assets_dir` 设置为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。
| 机器人 | 描述 | 资产 ID |
|-------|-------------|----------|
| ALOHA | 带平行夹持器的6-DoF双臂机器人 | trossen |
| 移动 ALOHA | 安装在 Slate 底座上的移动版 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 设置的带平行夹持器的7-DoF机械臂 | droid |
| Franka Emika (非 DROID) | 配备 Robotiq 2F-85 夹持器的 Franka FR3 机械臂 | franka |
| UR5e | 配备 Robotiq 2F-85 夹持器的6-DoF UR5e机械臂 | ur5e |
| UR5e 双手版 | 配备 Robotiq 2F-85 夹持器的双手 UR5e 配置 | ur5e_dual |
| ARX | 带平行夹持器的双手 ARX-5 机械臂配置 | arx |
| ARX 移动版 | 安装在 Slate 底座上的移动版双手 ARX-5 机械臂配置 | arx_mobile |
| Fibocom 移动版 | 配备 2x ARX-5 机械臂的 Fibocom 移动机器人 | fibocom_mobile |


## Pi0 模型动作空间定义

开箱即用，`pi0_base` 和 `pi0_fast_base` 使用以下动作空间定义（左右定义面向机器人从背后看向工作空间的方向）：
```
    "dim_0:dim_5": "左臂关节角度",
    "dim_6": "左臂夹持器位置",
    "dim_7:dim_12": "右臂关节角度（仅用于双手配置）",
    "dim_13": "右臂夹持器位置（仅用于双手配置）",

    # 对于移动机器人：
    "dim_14:dim_15": "xy 底座速度（仅用于移动机器人）",
```

本体感受状态使用与动作空间相同的定义，但移动机器人的底座 xy 位置（最后两个维度）不包含在本体感受状态中。

对于 7-DoF 机器人（如 Franka），我们使用动作空间的前 7 个维度进行关节动作，第 8 个维度用于夹持器动作。

Pi 机器人通用信息：
- 关节角度以弧度表示，位置零点对应于每个机器人接口库报告的零点位置，但 ALOHA 除外，标准 ALOHA 代码使用稍微不同的约定（详见 [ALOHA 示例代码](../examples/aloha_real/README.md)）。
- 夹持器位置范围为 [0.0, 1.0]，其中 0.0 对应完全打开，1.0 对应完全闭合。
- 控制频率为 UR5e 和 Franka 的 20 Hz，以及 ARX 和 Trossen (ALOHA) 机械臂的 50 Hz。

对于 DROID，我们使用原始的 DROID 动作配置，前 7 个维度为关节速度动作，第 8 个维度为夹持器动作，控制频率为 15 Hz。
