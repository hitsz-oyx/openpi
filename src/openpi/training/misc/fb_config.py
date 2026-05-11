import dataclasses
import einops
import numpy as np
import openpi.transforms as _transforms
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders


@dataclasses.dataclass(frozen=True)
class FurnitureInputs(_transforms.DataTransformFn):
    """Furniture Bench 输入转换 - 两个相机"""

    def __call__(self, data: dict) -> dict:
        base_image = np.asarray(data["observation/image"])
        wrist_image = np.asarray(data["observation/cam_wrist"])

        if base_image.shape[0] == 3:
            base_image = einops.rearrange(base_image, "c h w -> h w c")
        if wrist_image.shape[0] == 3:
            wrist_image = einops.rearrange(wrist_image, "c h w -> h w c")

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FurnitureOutputs(_transforms.DataTransformFn):
    """Furniture Bench 输出转换"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class LeRobotFurnitureDataConfig:
    repo_id: str
    default_prompt: str = None
    use_delta_joint_actions: bool = False
    assets_dir: str = None
    asset_id: str = "trossen"

    def create(self, assets_dirs, model_config):
        from openpi.training import config as _config
        from openpi.training.config import AssetsConfig, DataConfig
        
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "observation/image": "image",
                    "observation/cam_wrist": "wrist_image",
                    "observation/state": "state",
                    "actions": "action",
                })
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[FurnitureInputs()],
            outputs=[FurnitureOutputs()],
        )

        # 创建基本配置
        asset_id = self.asset_id or self.repo_id
        norm_stats = self._load_norm_stats(assets_dirs, asset_id, model_config)
        
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=asset_id,
            prompt_from_task=True,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            norm_stats=norm_stats,
            use_quantile_norm=model_config.model_type != _config.ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir, asset_id, model_config):
        from openpi.training.config import AssetsConfig
        
        if asset_id is None:
            return None
        try:
            from openpi import shared
            assets_config = AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            )
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = shared.normalize.load(shared.download.maybe_download(data_assets_dir))
            return norm_stats
        except FileNotFoundError:
            return None


def get_fb_config():
    """Get the Furniture Bench config."""
    from openpi.training.config import TrainConfig
    
    return TrainConfig(
        name="pi05_fb_low",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=8,
            action_horizon=16,
        ),
        data=LeRobotFurnitureDataConfig(
            repo_id="hitsz-oyx/furniture_bench_low",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=32,
    )
