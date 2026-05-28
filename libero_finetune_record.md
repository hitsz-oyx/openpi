LIBERO 数据集有几个相关链接：
| 用途 | 网址 |
|------|------|
| **LIBERO 官方网站** | https://libero-project.github.io/datasets |
| **原始 RLDS 格式（openvla 转换版）** | https://huggingface.co/datasets/openvla/modified_libero_rlds |
| **预转换 LeRobot 格式** | `physical-intelligence/libero`（HuggingFace 数据集） |

下载到 `assets/libero/` 的 tfrecord 文件就是从 openvla 那个链接下载的原始 RLDS 格式。

而 `pi05_libero` 配置使用的是 `physical-intelligence/libero`（LeRobot 格式），运行 `compute_norm_stats` 时会自动从 HuggingFace 拉取这个预转换版本。

下载到本地/home/u2023312616/test_ws/openpi/assets/libero
转换 RLDS → LeRobot 格式
cd /home/u2023312616/test_ws/openpi
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/u2023312616/test_ws/openpi/assets/libero
这会将 tfrecord 文件转换为 LeRobot 格式，并保存到 $HF_LEROBOT_HOME 目录，默认是~/.cache/huggingface/hub/

或者直接从官网下载 LeRobot 数据集
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 
uv run scripts/compute_norm_stats.py --config-name pi05_libero
会下载到~/.cache/huggingface/lerobot/physical-intelligence/libero/，并且生成归一化数据
