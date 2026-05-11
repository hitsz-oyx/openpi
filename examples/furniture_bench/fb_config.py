import dataclasses
import einops
import numpy as np
from openpi.training import config as _config
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig
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
class LeRobotFurnitureDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
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

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
        )


fb_config = TrainConfig(
    name="pi05_fb_low",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=16,
    ),
    data=LeRobotFurnitureDataConfig(
        repo_id="hitsz-oyx/furniture_bench_low",
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=50_000,
    batch_size=32,
)
