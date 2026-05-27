# Furniture Bench 微调指南（π₀.₅）

本文档描述如何使用 Furniture Bench 数据集对 π₀.₅ 模型进行微调训练。

---

## 环境

使用 **uv** 管理环境（Python 3.11）：

```bash
cd /home/u2023312616/test_ws/openpi
uv sync  # 创建 .venv
```

所有命令需通过 `uv run python ...` 执行。

---

## 数据集路径

- **本地数据**: `/home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low`
- **数据类型**: pickle 格式
- **数据类别**: cabinet, chair, desk, drawer, lamp, one_leg, round_table, square_table, stool
- **帧缓存**: 自动生成 `.frame_cache.json`（加速索引）

---

## 数据格式

Furniture Bench 数据集的 pickle 文件结构：

```python
{
    "furniture": str,       # 家具类型 e.g. "lamp"
    "observations": [       # 每个 step 的观测
        {
            "color_image1": np.ndarray,   # Wrist camera, shape (3, H, W) 或 (H, W, 3)
            "color_image2": np.ndarray,   # Front camera, shape (3, H, W) 或 (H, W, 3)
            "robot_state": dict,          # 见下方结构
        },
    ],
    "actions": np.ndarray,  # shape (T, 8), 8维动作
    "rewards": [...],
    "skills": [...],
    "success": bool,
}
```

### robot_state dict 结构

```python
robot_state = {
    "ee_pos": np.ndarray,      # (3,) end-effector position
    "ee_quat": np.ndarray,     # (4,) end-effector orientation quaternion
    "ee_pos_vel": np.ndarray,  # (3,) end-effector linear velocity
    "ee_ori_vel": np.ndarray,  # (3,) end-effector angular velocity
    "joint_positions": np.ndarray,  # (7,) 关节位置
    "joint_velocities": np.ndarray, # (7,) 关节速度
    "joint_torques": np.ndarray,    # (7,) 关节力矩
    "gripper_width": np.ndarray,    # (1,) 夹爪宽度
}
```

**8维状态向量** = `joint_positions(7) + gripper_width(1)`（拼接）

---

## 字段映射

| 原始字段 | 转换后字段 | 说明 |
|---|---|---|
| `obs["color_image2"]` | `observation/image` | Front（基座）相机 |
| `obs["color_image1"]` | `observation/cam_wrist` | Wrist（腕部）相机 |
| `obs["robot_state"]` (dict) | `observation/state` | 8维: joint_positions+gripper_width |
| `actions[i][:8]` | `actions` | 8维动作（action_horizon=16） |

**Image Naming**: `color_image2` → front camera (`base_0_rgb`), `color_image1` → wrist camera (`left_wrist_0_rgb`)

**Channel Format**: 数据中 image 为 CHW 格式 → 自动转换为 HWC 格式

---

## 模型配置

- **模型**: π₀.₅ (Pi0Config pi05=True)
- **动作维度**: 8
- **动作时域 (action_horizon)**: 16
- **语言模型**: Gemma 2B (LoRA)
- **动作专家**: Gemma 300M (LoRA)
- **训练精度**: bfloat16
- **优化器**: AdamW (weight_decay=1e-10, clip_norm=1.0)
- **LR Schedule**: warmup=1000, peak_lr=2.5e-05, decay=30000, end_lr=2.5e-06

---

## 配置文件

**主配置文件**: `src/openpi/training/misc/fb_config.py`

- Config Name: `pi05_fb_low`
- 预训练权重: `gs://openpi-assets/checkpoints/pi05_base/params`（通过 ActionDimWeightLoader）

### 资源路径

| 资源 | 路径 |
|---|---|
| Assets 目录 | `/home/u2023312616/test_ws/openpi/assets/pi05_fb_low` |
| 归一化统计量 | `assets/pi05_fb_low/furniture_bench_low/norm_stats.json` |
| Checkpoint 输出 | `checkpoints/pi05_fb_low/{exp_name}/` |

---

## 训练步骤

### 1. 运行训练

```bash
cd /home/u2023312616/test_ws/openpi

# 单 GPU 训练（测试模式，100 samples）
uv run python scripts/train_pytorch.py pi05_fb_low \
    --data.repo-id local_furniture_bench_low \
    --exp-name my_fb_experiment \
    --no-wandb-enabled \
    --overwrite

# 恢复训练
uv run python scripts/train_pytorch.py pi05_fb_low \
    --data.repo-id local_furniture_bench_low \
    --exp-name my_fb_experiment \
    --no-wandb-enabled \
    --resume
```

**注意**: `--data.repo-id` 是必需参数（因为 `DataConfigFactory.repo_id` 默认值为 `tyro.MISSING`）

### 2. 禁用 Wandb（如需）

wandb 可能因代理问题无法连接，使用 `--no-wandb-enabled`（tyro 使用 `--no-` 前缀禁用 bool 参数）。

### 3. 全量训练

编辑 `fb_config.py` 将 `max_samples = 100` 改为 `max_samples = None` 使用全部 ~86,100 帧数据。

---

## 计算归一化统计量（已完成）

统计量已预计算并存储在 `assets/pi05_fb_low/furniture_bench_low/norm_stats.json`。

如需重新计算：

```bash
cd /home/u2023312616/test_ws/openpi
uv run python scripts/compute_norm_stats.py pi05_fb_low \
    --data.repo-id local_furniture_bench_low
```

状态统计（8维，对应 joint_positions+gripper_width）：

| 维度 | mean | std |
|---|---|---|
| joint_0 | 0.5603 | 0.0729 |
| joint_1 | 0.0624 | 0.0695 |
| joint_2 | 0.0808 | 0.0418 |
| joint_3 | -0.5622 | 0.6892 |
| joint_4 | 0.0773 | 0.4258 |
| joint_5 | -0.0032 | 0.1352 |
| joint_6 | 0.0419 | 0.0348 |
| gripper | 0.0460 | 0.0222 |

---

## 数据流程

```
pickle file (furniture-bench raw data)
    ↓
FurniturePickleDataset
    - 帧索引缓存 (.frame_cache.json)
    - robot_state dict → 8D 向量 (joint_positions+gripper_width)
    - image CHW → HWC 转换
    - 构建 action_chunk (16, 8)
    ↓
__getitem__ 返回:
{
    "observation/image": front_image,     # HWC, uint8
    "observation/cam_wrist": wrist_image, # HWC, uint8
    "observation/state": state_vec,       # (8,), float32
    "action": action_chunk,              # (16, 8), float32
    "actions": action_chunk,             # (16, 8), float32
}
    ↓
data_transforms (Group):
  inputs=[
    FurnitureInputs(),       # 构建 image dict + prompt
    TokenizePrompt(...),     # tokenize prompt
  ]
  outputs=[
    FurnitureOutputs(),      # actions[:, :8]
  ]
    ↓
Normalize (使用 norm_stats.json 归一化 state 和 actions)
    ↓
CollatedObservationWrapper (batch 组装)
    ↓
Pi0Pytorch model (bfloat16, gradient checkpointing)
  - embed_image (SigLIP vision encoder)
  - embed_language_tokens (Gemma token embeddings)
  - PaliGemmaWithExpertModel.forward
    - prefix_lm + action_expert joint forward
    - compute_layer_complete (per-layer with grad checkpointing)
    - 输出 action tokens + action embeddings
  - action_mlp_head → action_pred
  - loss = flow_matching_loss(pred_action, target_action)
    ↓
loss.backward() → optim.step() → zero_grad()
```

---

## 已知问题与修复

### 1. CUDA OOM

**症状**: AdamW 优化器初始化 `exp_avg`/`exp_avg_sq` 时 OOM。

**解决方案**:
1. 移除 `model.float()` 转换，模型保持 bfloat16 精度（参数量减半）
2. 启用梯度检查点（gradient_checkpointing）
3. 反向传播后添加 `torch.cuda.empty_cache()`
4. 设置 `batch_size=1`

**显存占用对比**:

| 阶段 | float32 (原) | bfloat16 (优化后) |
|---|---|---|
| 模型创建后 | 14.47 GB | 7.48 GB |
| 反向传播后 | 27.51 GB | 14.26 GB |
| 稳定运行 | OOM | ~27.75 GB (7 GB 空闲) |

### 2. GemmaRMSNorm 返回值

自定义 `GemmaRMSNorm.forward` 始终返回 `(output, gate)` 元组。在 `compute_layer_complete` 中所有调用处需正确解包。

### 3. 梯度检查点兼容性

PyTorch 2.7.1 的 `torch.utils.checkpoint.checkpoint` 与 `use_reentrant=False` 配合使用时，需要确保自定义层返回正确格式。

---

## 修改配置

编辑 `src/openpi/training/misc/fb_config.py`:

```python
class LocalFurnitureDataConfigImpl(DataConfigFactory):
    local_data_dir: str = "/home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low"
    max_samples: int | None = 100  # None = 全量训练
    assets: AssetsConfig = dataclasses.field(default_factory=lambda: AssetsConfig(
        assets_dir="/home/u2023312616/test_ws/openpi/assets/pi05_fb_low",
        asset_id="furniture_bench_low",
    ))
```

---

## 验证训练

训练开始后，预期输出：

```
Found 100 pickle files in /home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low
Loaded norm stats from /home/u2023312616/test_ws/openpi/assets/pi05_fb_low/furniture_bench_low
Model created with default dtype (config: bfloat16)
Gradient checkpointing enabled
Step 0 (after_model_creation): GPU memory - allocated: 7.48GB
step=0 loss=2.2563 lr=2.50e-08 grad_norm=90.89 time=3.6s
```

训练速度: ~1.3 it/s (batch_size=1, 单 GPU)