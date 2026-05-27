"""计算配置的归一化统计信息。

此脚本用于计算给定配置的归一化统计信息。它将计算数据集中数据的
均值、标准差以及分位数，并保存到配置的 assets 目录中。
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """移除字符串类型的字段。

    由于 JAX 不支持字符串类型，且计算归一化统计信息时不需要字符串，
    因此将其移除。
    """
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建 PyTorch 数据加载器（用于 LeRobot 格式数据集）。

    Args:
        data_config: 数据配置
        action_horizon: 动作视野，用于创建数据集
        batch_size: 批大小
        model_config: 模型配置
        num_workers: 数据加载的工作进程数
        max_frames: 最大帧数限制（用于快速测试）

    Returns:
        数据加载器和批次数
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # 移除字符串，因为 JAX 不支持且计算归一化统计信息时不需要
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建 RLDS 数据加载器（用于 RLDS 格式数据集）。

    Args:
        data_config: 数据配置
        action_horizon: 动作视野
        batch_size: 批大小
        max_frames: 最大帧数限制

    Returns:
        数据加载器和批次数
    """
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # 移除字符串
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # 注意：此长度目前是针对 DROID 硬编码的
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """
    主函数：计算并保存归一化统计信息。

    Args:
        config_name: 配置名称（如 pi05_libero）
        max_frames: 可选的帧数限制，用于快速测试
    """
    # 获取配置
    config = _config.get_config(config_name)
    # 创建数据配置
    data_config = config.data.create(config.assets_dirs, config.model)

    # 根据数据类型选择数据加载器（LeRobot 格式 vs RLDS 格式）
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 需要计算归一化统计的键：state（本体感受）和 actions（动作）
    keys = ["state", "actions"]
    # 使用 RunningStats 计算滑动统计信息
    stats = {key: normalize.RunningStats() for key in keys}

    # 遍历数据，更新统计信息
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    # 获取最终统计信息
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 保存到 assets 目录
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
