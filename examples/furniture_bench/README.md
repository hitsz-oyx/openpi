# Furniture Bench 微调指南

此目录包含 Furniture Bench 数据集适配 OpenPi 并进行微调的必要文件。

## 数据集信息

Furniture Bench 原始数据集包含：

- **两个相机**：`color_image1`（主相机）和 `color_image2`（腕部相机）
- **状态维度**：8维 `[pos(3), quat(4), gripper_width(1)]`
- **动作维度**：8维 `[pos_delta(3), ori_delta(3), gripper(2)]`

## 配置文件

| 文件                              | 说明                      |
| ------------------------------- | ----------------------- |
| `fb_config.py`                  | 训练配置文件，定义了模型、数据转换、权重加载等 |
| `convert_fb_optimized.py`       | 数据格式转换脚本（支持并行加速）      |
| `compute_norm_stats_fb.py`      | 归一化统计量计算脚本             |

## 环境配置

### 方法一：使用独立的 Conda 环境（推荐）

创建一个专门用于数据转换的环境：

```bash
# 创建环境
conda create -n fb_convert python=3.10 -y

# 激活环境
conda activate fb_convert

# 安装依赖
pip install numpy>=1.24 safetensors>=0.4.0 pillow>=10.0 tyro>=0.5.0 tqdm>=4.66.0 huggingface-hub>=0.20.0
```

### 方法二：使用项目自带的依赖文件

```bash
cd /home/u2023312616/test_ws/openpi
pip install -r examples/furniture_bench/requirements.txt
```

## 微调步骤

### 1. 数据转换

使用优化的转换脚本（支持并行加速，不依赖 lerobot 包）：

```bash
conda activate fb_convert

# 并行模式（默认，4个进程）
python examples/furniture_bench/convert_fb_optimized.py \
    --data-dir /home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low \
    --repo-id hitsz-oyx/furniture_bench_low \
    --fps 10 \
    --num-workers 4

# 顺序模式（禁用并行）
python examples/furniture_bench/convert_fb_optimized.py \
    --data-dir /home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low \
    --repo-id hitsz-oyx/furniture_bench_low \
    --no-parallel
```

**参数说明：**

| 参数           | 说明                              | 默认值                                                                  |
| ------------ | ------------------------------- | -------------------------------------------------------------------- |
| `--data-dir` | 原始 `.pkl` 数据目录                  | `/home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low` |
| `--repo-id`  | 转换后数据集的名称（会作为存储目录名）             | `hitsz-oyx/furniture_bench_low`                                      |
| `--fps`      | **数据采集帧率**（非动作预测长度），表示每秒采集多少帧数据 | `10`                                                                 |
| `--num-workers` | 并行进程数（仅在启用并行时生效）           | `4`                                                                  |
| `--parallel` / `--no-parallel` | 是否启用并行处理                        | 默认为启用                                                              |

**fps 参数说明：**

你说的没错！数据集里的数据确实是一帧一帧的，`fps` 参数**不会改变数据内容或做采样**。它的主要作用是：

1. **元数据记录**：作为数据集的元信息存储，告诉下游系统数据的时间分辨率
2. **时间戳计算**：LeRobot 内部会根据 `fps` 为每帧生成时间戳（如第0帧=0.0秒，第1帧=0.1秒，第2帧=0.2秒...）
3. **下游任务使用**：某些模型或评估脚本可能需要知道帧率来正确处理时间相关的操作

**关键点：**
- **不会丢弃帧**：转换时会保留所有原始帧，不会因为 `fps=10` 而采样或丢弃数据
- **不影响动作预测**：与 `action_horizon`（模型一次预测多少步动作）完全无关
- **建议值**：原始 Furniture Bench 数据集的采集帧率为10fps，建议保持此值以保持时间戳的准确性

**fps 在数据转换中的实际效果：**
```
假设原始数据有100帧：
- fps=10 → 时间戳: 0.0, 0.1, 0.2, ..., 9.9 秒
- fps=20 → 时间戳: 0.0, 0.05, 0.1, ..., 4.95 秒

数据内容完全相同，只是时间戳不同！
```

**转换后文件存储位置：**

```
~/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low/
```

完整路径示例（假设用户名是 `u2023312616`）：

```
/home/u2023312616/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low/
```

### 2. 配置训练参数

编辑 `fb_config.py` 中的配置：

```python
fb_config = TrainConfig(
    name="pi05_fb_low",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,        # Furniture Bench 动作为8维
        action_horizon=16,   # 动作预测序列长度
    ),
    data=LeRobotFurnitureDataConfig(
        repo_id="hitsz-oyx/furniture_bench_low",  # 与转换脚本一致
        base_config=DataConfig(
            prompt_from_task=True,  # 从任务生成prompt
        ),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=50_000,  # 训练步数
    batch_size=32,           # 批量大小（根据显存调整）
)
```

### 3. 计算归一化统计量

使用目录下的专用脚本计算归一化统计量：

```bash
python examples/furniture_bench/compute_norm_stats_fb.py \
    --data-dir /home/u2023312616/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low \
    --output-dir /home/u2023312616/test_ws/openpi/assets/hitsz-oyx/furniture_bench_low
```

**参数说明：**

| 参数           | 说明                              | 默认值                                                                  |
| ------------ | ------------------------------- | -------------------------------------------------------------------- |
| `--data-dir` | 转换后的 safetensors 数据目录           | `/home/u2023312616/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low` |
| `--output-dir` | 统计量输出目录                        | `/home/u2023312616/test_ws/openpi/assets/hitsz-oyx/furniture_bench_low` |
| `--max-frames` | 最大处理帧数（用于测试，None 表示全部）        | `None`                                                                |

### 4. 开始训练

**完整训练命令：**

```bash
# 激活环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate openpi_server_eval

# 设置 PYTHONPATH
export PYTHONPATH=/home/u2023312616/test_ws/openpi/src:$PYTHONPATH

# 进入项目目录
cd /home/u2023312616/test_ws/openpi

# 开始训练（测试模式，仅训练10步）
python scripts/train.py pi05_fb_low --exp-name fb_finetune_test --num-train-steps 10 --overwrite

# 正式训练（50,000步）
# python scripts/train.py pi05_fb_low --exp-name fb_finetune --num-train-steps 50000
```

**关键配置说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `action_dim` | 8 | Furniture Bench 动作维度 |
| `action_horizon` | 16 | 动作预测序列长度 |
| `batch_size` | 32 | 批量大小（根据显存调整） |
| `num_train_steps` | 50000 | 训练步数 |

**动作维度适配：**

由于预训练模型 `pi05_base` 是基于 32 维动作训练的，而 Furniture Bench 只有 8 维动作，我们使用了自定义的 `ActionDimWeightLoader` 来自动适配权重维度：

```python
weight_loader=weight_loaders.ActionDimWeightLoader(
    params_path="gs://openpi-assets/checkpoints/pi05_base/params",
    target_action_dim=8,
)
```

**注意事项：**

1. 如果遇到显存不足的问题，可以尝试：
   - 减小 `batch_size`（在 `fb_config.py` 中修改）
   - 使用更小的模型配置
   - 在具有更多显存的 GPU 上运行

2. 训练第一次运行时会下载约 11.6 GB 的预训练模型权重，请确保网络连接稳定

3. 训练过程会自动上传日志到 Weights & Biases，确保已登录：
   ```bash
   wandb login
   ```

## 数据流程

```
原始数据 (pkl)          转换后 (LeRobot)        模型输入 (fb_config)
├── color_image1    →   observation/image   →   base_0_rgb       (主相机)
├── color_image2    →   observation/cam_wrist →  left_wrist_0_rgb (腕部相机)
└── robot_state     →   observation/state   →   state            (8维)
```

## 数据存储说明

### 数据转换方式

**数据是预转换并存储到磁盘的，训练时不会实时转换。**

转换脚本会将原始 `.pkl` 文件转换为 LeRobot 格式，并存储到默认目录：

```
~/.cache/huggingface/datasets/lerobot/<repo_id>
```

例如 `repo_id="hitsz-oyx/furniture_bench_low"` 会存储到：

```
~/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low
```

### 为什么需要预转换

1. **性能优化**：避免训练时重复解析 `.pkl` 文件
2. **格式统一**：转换为 LeRobot 标准格式，便于数据加载和处理
3. **数据缓存**：转换一次后可以多次使用

### 数据流程示意图

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 数据转换阶段 (一次性)                                        │
│                                                                 │
│   raw_data/*.pkl  ──convert_fb_to_lerobot.py──>  LeRobot格式    │
│                                                    │            │
│                                                    v            │
│                                            ~/.cache/huggingface/│
│                                                datasets/lerobot/│
│                                               hitsz-oyx___fb/  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 训练时读取
                              v
┌─────────────────────────────────────────────────────────────────┐
│ 2. 训练阶段 (多次)                                              │
│                                                                 │
│   train.py  <──read──  LeRobot格式数据                          │
│     │                                                           │
│     v                                                           │
│   模型训练                                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 如何验证转换是否成功

转换完成后，可以检查输出目录是否存在：

```bash
ls ~/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low
```

## 注意事项

1. 确保 `convert_fb_optimized.py` 中的 `repo_id` 与 `fb_config.py` 中的 `repo_id` 一致
2. 如果遇到显存不足，减小 `batch_size`
3. `action_dim=8` 是 Furniture Bench 的原始动作维度，无需修改
4. 数据转换时会将两个相机的图像都保存到 LeRobot 数据集中
5. 如果修改了转换脚本，需要重新运行转换并删除旧的缓存数据
6. 并行转换时，建议根据 CPU 核心数调整 `--num-workers` 参数（默认4）

