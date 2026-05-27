"""
Furniture Bench 数据集配置模块

本模块用于配置 Furniture Bench 机器人数据集的加载和预处理流程。
支持直接从本地 pickle 格式读取原始数据，无需预先转换为 LeRobot 格式。

数据集结构:
    - 原始数据: furniture-bench 导出的 .pkl 文件，包含 observation/action 数据
    - 相机: front_camera (color_image2) 和 wrist_camera (color_image1)
    - 状态: 8维向量 [joint_positions(7), gripper_width(1)]
    - 动作: 8维向量 [pos_delta(3), ori_delta(3), gripper(2)]
"""
import dataclasses
import pathlib
from typing import TYPE_CHECKING

import einops
import numpy as np

import openpi.transforms as _transforms
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders

if TYPE_CHECKING:
    from openpi.training.config import DataConfig


# =============================================================================
# 数据预处理 Transform
# =============================================================================

@dataclasses.dataclass(frozen=True)
class FurnitureInputs(_transforms.DataTransformFn):
    """
    Furniture Bench 输入数据转换

    将原始的 observation 数据转换为模型期望的格式:
    - observation/image       -> base_0_rgb (主相机)
    - observation/cam_wrist  -> left_wrist_0_rgb (腕部相机)
    - observation/state       -> state (机器人状态)

    注意: right_wrist_0_rgb 设为全零，因为 Furniture Bench 只有一个腕部相机
    """

    def __call__(self, data: dict) -> dict:
        # -------------------------------------------------------------------------
        # 1. 提取图像数据
        # -------------------------------------------------------------------------
        # 从 data 字典中提取主相机图像 (front camera, 对应 pickle 中的 color_image2)
        base_image = np.asarray(data["observation/image"])
        # 从 data 字典中提取腕部相机图像 (wrist camera, 对应 pickle 中的 color_image1)
        wrist_image = np.asarray(data["observation/cam_wrist"])

        # -------------------------------------------------------------------------
        # 2. 处理图像通道顺序
        # -------------------------------------------------------------------------
        # 如果图像是 (C, H, W) 格式 (通道在前)，转换为 (H, W, C) 格式 (通道在后)
        # 这是因为 pickle 中某些图像可能是 CHW 格式，而模型期望 HWC 格式
        if base_image.shape[0] == 3:
            base_image = einops.rearrange(base_image, "c h w -> h w c")
        if wrist_image.shape[0] == 3:
            wrist_image = einops.rearrange(wrist_image, "c h w -> h w c")

        # -------------------------------------------------------------------------
        # 3. 构建模型输入格式
        # -------------------------------------------------------------------------
        # image 字典的 key 必须是模型定义的三个标准相机 key:
        #   - base_0_rgb: 主相机视角
        #   - left_wrist_0_rgb: 左腕部相机
        #   - right_wrist_0_rgb: 右腕部相机 (Furniture Bench 没有，用全零图填充)
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,           # 主相机图像
                "left_wrist_0_rgb": wrist_image,    # 腕部相机图像
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 全零填充
            },
            # image_mask 指定哪些相机图像是有效的 (True=有效, False=无效/全零)
            # base_0_rgb 和 left_wrist_0_rgb 都有有效图像，right_wrist_0_rgb 是全零图所以为 False
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": "Assemble the furniture",  # 默认语言提示
        }

        # -------------------------------------------------------------------------
        # 4. 传递动作数据 (如果有)
        # -------------------------------------------------------------------------
        # actions 字段在训练时需要，推理时可能没有
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 如果有自定义 prompt，覆盖默认 prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FurnitureOutputs(_transforms.DataTransformFn):
    """
    Furniture Bench 输出数据转换

    将原始动作数据转换为模型输出的期望格式。
    由于模型输出的是完整动作序列，这里只取前 8 维 (Furniture Bench 动作维度)。
    """

    def __call__(self, data: dict) -> dict:
        # data["actions"] 形状为 (action_horizon, action_dim)
        # 取前 8 维动作 (pos_delta(3) + ori_delta(3) + gripper(2))
        return {"actions": np.asarray(data["actions"][:, :8])}


# =============================================================================
# 数据集加载器
# =============================================================================

def _create_furniture_data_config():
    """
    创建 Furniture Bench 数据配置

    该函数返回一个 DataConfigFactory 实现类，用于:
    1. 直接加载本地 pickle 格式的原始数据 (无需预转换)
    2. 配置数据增强和预处理流程
    3. 加载归一化统计量

    内部类:
        - FurniturePickleDataset: 直接读取 pickle 格式数据的 dataset
        - LocalFurnitureDataConfigImpl: 数据配置工厂类
    """
    import pickle
    from typing import Iterator
    import tyro
    import numpy as np

    from openpi.training.data_loader import Dataset
    from openpi.models.model import BaseModelConfig

    import openpi.training.config as _config
    from openpi.training.config import DataConfigFactory, AssetsConfig, DataConfig
    from openpi.models.tokenizer import PaligemmaTokenizer

    class FurniturePickleDataset(Dataset):
        """
        直接加载本地 furniture-bench pickle 格式数据集

        Furniture Bench 原始数据存储为 .pkl 文件，每个文件包含一个 episode 的数据。
        该类负责:
        1. 扫描数据目录，收集所有 .pkl 文件
        2. 建立帧索引，支持随机访问任意帧
        3. 缓存帧索引以加速后续加载

        数据格式 (每个 .pkl 文件):
            - observations: list of dict, 每个 dict 包含:
                - color_image1: wrist camera 图像 (H, W, 3)
                - color_image2: front camera 图像 (H, W, 3)
                - robot_state: dict, 包含 joint_positions 和 gripper_width
            - actions: list of array, 每个 action 是 8 维向量

        注意: color_image1 对应腕部相机, color_image2 对应主相机
        """

        def __init__(
            self,
            data_dir: pathlib.Path,
            action_horizon: int = 16,
            max_frames: int | None = None,
        ):
            """
            初始化数据集

            Args:
                data_dir: pickle 数据文件所在目录
                action_horizon: 动作预测序列长度 (默认 16)
                max_frames: 最大帧数限制 (用于调试, None 表示不限制)
            """
            self.data_dir = pathlib.Path(data_dir)
            self.action_horizon = action_horizon

            # -------------------------------------------------------------------------
            # 1. 扫描并收集所有 pickle 文件
            # -------------------------------------------------------------------------
            # 数据目录结构: data_dir/category/*.pkl
            # 每个 category 是一个子目录，包含该类别的所有 episode
            self.pkl_files = []
            for category_dir in sorted(self.data_dir.iterdir()):
                if category_dir.is_dir():
                    pkl_files = sorted(category_dir.glob("*.pkl"))
                    self.pkl_files.extend(pkl_files)

            # 如果设置了 max_frames，只使用前 max_frames 个文件
            if max_frames is not None and max_frames > 0:
                self.pkl_files = self.pkl_files[:max_frames]

            print(f"Found {len(self.pkl_files)} pickle files in {data_dir}")

            # -------------------------------------------------------------------------
            # 2. 构建帧索引 (累计帧数)
            # -------------------------------------------------------------------------
            # 为了支持 O(1) 随机访问，需要知道每个文件包含多少帧
            # _cumulative_frames[i] 表示前 i+1 个文件包含的总帧数
            # 例如: [100, 250, 380] 表示第0个文件有100帧，第1个文件有150帧...
            cache_file = self.data_dir / ".frame_cache.json"
            self._cumulative_frames = []

            # 尝试从缓存文件加载索引
            if cache_file.exists():
                import json
                with open(cache_file) as f:
                    cached = json.load(f)
                cached_files = cached.get("files", [])
                cached_paths = [pathlib.Path(p) for p in cached_files]
                # 只有当文件列表没变时才使用缓存
                if cached_paths == self.pkl_files:
                    self._cumulative_frames = cached["cumulative"]
                    print(f"Loaded frame cache from {cache_file} ({len(self._cumulative_frames)} files)")

            # 缓存不存在或失效，需要重新索引
            if not self._cumulative_frames:
                total = 0
                for i, pkl_file in enumerate(self.pkl_files):
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)
                    total += len(data["observations"])  # 累加当前文件的帧数
                    self._cumulative_frames.append(total)
                    if (i + 1) % 500 == 0 or i == len(self.pkl_files) - 1:
                        print(f"Indexed {i+1}/{len(self.pkl_files)} files ({total} frames)")
                # 保存索引缓存
                if cache_file.parent.exists():
                    import json
                    with open(cache_file, "w") as f:
                        json.dump({
                            "cumulative": self._cumulative_frames,
                            "files": [str(p) for p in self.pkl_files],
                        }, f)
                    print(f"Saved frame cache to {cache_file}")

        def _find_file_and_frame(self, index: int) -> tuple[int, int, dict]:
            """
            根据全局帧索引找到对应的文件和帧号

            Args:
                index: 全局帧索引 (0 到总帧数-1)

            Returns:
                tuple: (帧在文件内的索引, 该文件总帧数, 该文件的数据字典)
            """
            # 找到第一个 cum_count > index 的位置，即目标文件
            for file_idx, cum_count in enumerate(self._cumulative_frames):
                if index < cum_count:
                    # 计算在文件内的帧索引
                    prev = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
                    frame_idx = index - prev
                    # 加载该文件的数据
                    traj_data = self._load_trajectory(self.pkl_files[file_idx])
                    return frame_idx, len(traj_data["observations"]), traj_data
            raise IndexError(f"Index {index} out of range")

        def _load_trajectory(self, pkl_file: pathlib.Path) -> dict:
            """加载单个 episode 的数据"""
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            return data

        def __getitem__(self, index: int) -> dict:
            """
            获取指定索引的样本

            Returns:
                dict: 包含以下字段:
                    - observation/image: 主相机图像 (H, W, 3)
                    - observation/cam_wrist: 腕部相机图像 (H, W, 3)
                    - observation/state: 机器人状态 (8,)
                    - action: 动作序列 (action_horizon, 8)
                    - actions: 同 action (兼容性别名)
            """
            # -------------------------------------------------------------------------
            # 1. 定位到目标帧
            # -------------------------------------------------------------------------
            frame_idx, num_obs, traj_data = self._find_file_and_frame(index)

            # -------------------------------------------------------------------------
            # 2. 提取当前帧的 observation
            # -------------------------------------------------------------------------
            observations = traj_data["observations"]
            actions = traj_data["actions"]
            num_actions = len(actions)
            action_dim = 8  # Furniture Bench 动作维度

            obs = observations[frame_idx]
            # 注意: Furniture Bench 的 color_image1 是腕部相机, color_image2 是主相机
            front_image = obs["color_image2"]   # 主相机
            wrist_image = obs["color_image1"]    # 腕部相机
            robot_state = obs["robot_state"]

            # -------------------------------------------------------------------------
            # 3. 处理图像格式
            # -------------------------------------------------------------------------
            # 某些 pickle 文件中图像可能是 (3, H, W) 格式，需要转换为 (H, W, 3)
            if isinstance(front_image, np.ndarray) and front_image.ndim == 3 and front_image.shape[0] == 3:
                front_image = np.moveaxis(front_image, 0, -1)
            if isinstance(wrist_image, np.ndarray) and wrist_image.ndim == 3 and wrist_image.shape[0] == 3:
                wrist_image = np.moveaxis(wrist_image, 0, -1)

            # -------------------------------------------------------------------------
            # 4. 构建状态向量
            # -------------------------------------------------------------------------
            # robot_state 格式: dict 包含 joint_positions 和 gripper_width
            # 拼接为 8 维向量: [joint(7), gripper(1)]
            if isinstance(robot_state, dict):
                state_vec = np.concatenate([
                    np.asarray(robot_state["joint_positions"]).ravel(),
                    np.asarray(robot_state["gripper_width"]).ravel(),
                ]).astype(np.float32)
            else:
                state_vec = np.asarray(robot_state).ravel().astype(np.float32)

            # -------------------------------------------------------------------------
            # 5. 构建动作序列 (action chunk)
            # -------------------------------------------------------------------------
            # 动作序列从当前帧开始，连续取 action_horizon 帧
            # 如果超出 episode 范围，用零填充
            action_horizon = self.action_horizon
            action_chunk = np.zeros((action_horizon, action_dim), dtype=np.float32)
            for i in range(action_horizon):
                idx = frame_idx + i
                if idx < num_actions:
                    action_chunk[i] = actions[idx][:action_dim]

            # -------------------------------------------------------------------------
            # 6. 返回标准化的数据格式
            # -------------------------------------------------------------------------
            return {
                "observation/image": front_image,      # 主相机 -> base_0_rgb
                "observation/cam_wrist": wrist_image,  # 腕部相机 -> left_wrist_0_rgb
                "observation/state": state_vec,         # 状态向量
                "action": action_chunk,                 # 动作序列 (旧字段名)
                "actions": action_chunk,                # 动作序列 (新字段名)
            }

        def __len__(self) -> int:
            """返回数据集中的总帧数"""
            return self._cumulative_frames[-1] if self._cumulative_frames else 0

    class LocalFurnitureDataConfigImpl(DataConfigFactory):
        """
        Furniture Bench 本地数据配置工厂

        该类实现 DataConfigFactory 接口，用于创建完整的 DataConfig 配置对象。
        主要职责:
        1. 创建 FurniturePickleDataset 数据集实例
        2. 配置数据预处理流程 (FurnitureInputs, TokenizePrompt, FurnitureOutputs)
        3. 加载归一化统计量
        """

        use_delta_joint_actions: bool = False  # 是否使用增量关节动作 (Furniture Bench 不需要)
        default_prompt: str | None = None      # 默认 prompt (None 表示使用 FurnitureInputs 中的默认)

        # 数据集标识
        repo_id: str = "local_furniture_bench_low"

        # 资产配置 (包含 tokenizer 和归一化统计量)
        assets: AssetsConfig = dataclasses.field(default_factory=lambda: AssetsConfig(
            assets_dir="/home/u2023312616/test_ws/openpi/assets/pi05_fb_low",
            asset_id="furniture_bench_low",
        ))

        # 基础配置
        base_config: tyro.conf.Suppress[DataConfig | None] = dataclasses.field(
            default_factory=lambda: DataConfig(
                repo_id="local_furniture_bench_low",
                prompt_from_task=True,  # 从任务生成 prompt
            )
        )

        max_samples: int | None = None  # 最大样本数限制 (None 表示不限制)

        # 本地数据目录
        local_data_dir: str = "/home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low"

        # 内部使用的配置 (用于 ensure 正确的值)
        _asset_id: str = "furniture_bench_low"
        _assets_dir: str = "/home/u2023312616/test_ws/openpi/assets/pi05_fb_low"

        def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
            """
            创建 DataConfig 配置对象

            Args:
                assets_dirs: 资产目录路径
                model_config: 模型配置

            Returns:
                DataConfig: 包含数据集、数据变换、归一化统计量等完整配置
            """
            import etils.epath as epath
            from openpi.training import data_loader as _data
            from openpi.training.config import ModelTransformFactory

            # -------------------------------------------------------------------------
            # 1. 配置数据预处理流程
            # -------------------------------------------------------------------------
            # 数据变换顺序:
            #   RepackTransform (已设置为空，这里不做任何 repack)
            #   -> FurnitureInputs (图像/状态转换)
            #   -> TokenizePrompt (prompt tokenization)
            #   -> Normalize (归一化)
            #   -> 模型特定的 transform
            data_transforms = _transforms.Group(
                inputs=[
                    FurnitureInputs(),                           # 图像/状态格式转换
                    _transforms.TokenizePrompt(PaligemmaTokenizer())  # prompt token化
                ],
                outputs=[FurnitureOutputs()],  # 输出变换 (取动作前8维)
            )

            # -------------------------------------------------------------------------
            # 2. 加载归一化统计量
            # -------------------------------------------------------------------------
            asset_id = self._asset_id
            assets_dir = self._assets_dir
            norm_stats = self._load_norm_stats(epath.Path(assets_dir), asset_id)

            # -------------------------------------------------------------------------
            # 3. 创建数据集实例
            # -------------------------------------------------------------------------
            dataset = FurniturePickleDataset(
                data_dir=self.local_data_dir,
                action_horizon=16,  # 动作预测序列长度
                max_frames=self.max_samples,
            )

            # 如果设置了 max_samples，包装为 LimitedDataset
            limited_dataset = _data.LimitedDataset(dataset, self.max_samples) if self.max_samples else dataset

            # 模型 transform (ResizeImages + PadStatesAndActions)
            model_transforms = ModelTransformFactory()(model_config)

            # -------------------------------------------------------------------------
            # 4. 返回完整配置
            # -------------------------------------------------------------------------
            return DataConfig(
                repo_id=self.repo_id,
                asset_id=asset_id,
                prompt_from_task=True,
                repack_transforms=_transforms.Group(),  # 空 repack，因为我们直接返回正确格式
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                norm_stats=norm_stats,
                use_quantile_norm=True,  # pi05 使用分位数归一化
                local_dataset=dataset,    # 直接传递 dataset 对象
            )

    return LocalFurnitureDataConfigImpl()


def get_fb_config():
    """
    获取 Furniture Bench 训练配置

    Returns:
        TrainConfig: 包含完整的训练配置，包括:
            - 模型配置 (pi0.5, 8维动作, 16步动作预测)
            - 数据配置 (本地 pickle 数据)
            - 权重加载器 (支持 32维->8维 动作维度适配)
            - 训练参数 (5000步, batch_size=1)
    """
    from openpi.training.config import TrainConfig

    # -------------------------------------------------------------------------
    # 模型配置: Pi0.5 (轻量级版本)
    # -------------------------------------------------------------------------
    model_config = pi0_config.Pi0Config(
        pi05=True,                           # 使用 pi0.5 模型
        action_dim=8,                         # Furniture Bench 动作维度
        action_horizon=16,                    # 每次预测 16 步动作
        discrete_state_input=False,            # 状态作为连续输入而非离散 token
        paligemma_variant="gemma_2b_lora",    # 视觉编码器 variant
        action_expert_variant="gemma_300m_lora",  # 动作专家 variant
    )

    # -------------------------------------------------------------------------
    # 构建完整训练配置
    # -------------------------------------------------------------------------
    return TrainConfig(
        name="pi05_fb_low",
        model=model_config,
        data=_create_furniture_data_config(),  # 数据配置 (本地 pickle 格式)
        weight_loader=weight_loaders.ActionDimWeightLoader(
            # 预训练权重路径 (32维动作)
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
            target_action_dim=8,  # 目标动作维度 (8维 Furniture Bench)
        ),
        num_train_steps=5_000,   # 训练步数
        batch_size=1,            # 批量大小 (调试时用小值)
        freeze_filter=model_config.get_freeze_filter(),  # 冻结过滤器
        ema_decay=None,         # EMA 衰减 (None 表示不使用 EMA)
    )
