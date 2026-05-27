import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """创建一个随机的 Libero 策略输入示例。"""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """
    解析图像数据，转换为正确的格式 (H, W, C) 和 uint8 类型。

    LeRobot 自动将图像存储为 float32 (C, H, W) 格式，
    而此函数将其转换为模型期望的 uint8 (H, W, C) 格式。
    """
    image = np.asarray(image)
    # 如果是浮点类型，转换为 uint8 (范围 0-255)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # 如果是 (C, H, W) 格式，转换为 (H, W, C) 格式
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
   此类用于将输入数据转换为模型期望的格式，同时适用于训练和推理。

    对于你自己的数据集，你可以复制此类，并根据下面的注释修改键名，
    以便将数据集中的正确元素传递给模型。
    """

    # 指定要使用的模型类型
    # 对于你自己的数据集，请不要修改此值
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 可能需要将图像解析为 uint8 (H,W,C) 格式
        # 因为 LeRobot 自动存储为 float32 (C,H,W)，但策略推理时会跳过此处理。
        # 保留此逻辑用于你自己的数据集，但如果你的数据集将图像存储在
        # 与 "observation/image" 或 "observation/wrist_image" 不同的键中，
        # 你应该在下面修改它。

        # Pi0 模型目前支持三种图像输入：一种第三人称视角，
        # 和两种手腕视角（左侧和右侧）。如果你的数据集没有特定类型的图像，
        # 例如手腕图像，你可以在这里将其注释掉，并用零数组替换，
        # 就像我们对右侧手腕图像的处理一样。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建输入字典。请勿更改下面字典中的键名。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,          # 第三人称视角图像
                "left_wrist_0_rgb": wrist_image,  # 左手腕图像
                # 用适当形状的零数组填充任何不存在的图像
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 我们只为 pi0 模型掩码填充图像，而不是 pi0-FAST。
                # 对于你自己的数据集，请勿修改此值。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 填充动作到模型动作维度。保留此逻辑用于你自己的数据集。
        # 动作仅在训练期间可用。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将提示（又称语言指令）传递给模型。
        # 保留此逻辑用于你自己的数据集（但如果指令不是存储在 "prompt" 中，
        # 则修改键名；输出字典始终需要包含 "prompt" 键）。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    此类用于将模型输出转换回数据集特定格式。仅用于推理。

    对于你自己的数据集，你可以复制此类，并根据下面的注释修改动作维度。
    """

    def __call__(self, data: dict) -> dict:
        # 仅返回前 N 个动作——由于我们上面填充动作以适应模型动作维度，
        # 我们现在需要从返回字典中解析出正确数量的动作。
        # 对于 Libero，我们只返回前 7 个动作（因为其余的是填充）。
        # 对于你自己的数据集，将 `7` 替换为你的数据集的动作维度。
        return {"actions": np.asarray(data["actions"][:, :7])}
