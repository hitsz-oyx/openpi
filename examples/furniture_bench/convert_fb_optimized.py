#!/usr/bin/env python3
"""
优化的数据转换脚本，支持并行处理加速。
直接从 Furniture Bench 的 pkl 文件生成 LeRobot 格式的数据集。

依赖：
- numpy
- safetensors
- pillow
- tyro
- tqdm
- huggingface-hub
"""
import sys
import os
import pickle
from pathlib import Path
import numpy as np
import tyro
from tqdm import tqdm
from safetensors.numpy import save_file
from concurrent.futures import ProcessPoolExecutor, as_completed


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_frame(output_dir, frame_idx, data):
    """保存单帧数据为 safetensors 文件"""
    filename = f"frame_{frame_idx}.safetensors"
    save_file(data, str(output_dir / filename))


def process_episode(args):
    """处理单个 episode 文件（用于并行处理）"""
    pkl_file, fps, start_frame_idx, output_dir = args
    frames_written = 0
    
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        observations = data["observations"]
        actions = np.array(data["actions"])
        
        for i, obs in enumerate(observations):
            img_main = obs["color_image1"]
            img_wrist = obs["color_image2"]
            
            if img_main.dtype != np.uint8:
                img_main = (img_main * 255).astype(np.uint8)
            if img_wrist.dtype != np.uint8:
                img_wrist = (img_wrist * 255).astype(np.uint8)
            
            rs = obs["robot_state"]
            pos = rs["ee_pos"]
            quat = rs["ee_quat"]
            gripper = rs["gripper_width"]
            
            state = np.concatenate([pos, quat, gripper]).astype(np.float32)
            
            act_idx = min(i, len(actions) - 1)
            action = actions[act_idx].astype(np.float32)
            
            timestamp = np.float32((start_frame_idx + i) / fps)
            
            frame_data = {
                "image": img_main,
                "cam_wrist": img_wrist,
                "state": state,
                "action": action,
                "timestamp": timestamp,
                "episode_index": np.int64(0),
                "frame_index_in_episode": np.int64(i),
            }
            
            save_frame(output_dir, start_frame_idx + i, frame_data)
            frames_written += 1
        
        return frames_written
    
    except Exception as e:
        print(f"处理 {pkl_file} 时出错: {e}", file=sys.stderr)
        return 0


def convert_furniture_bench_to_lerobot(
    data_dir: str = "/home/u2023312616/test_ws/furniture-bench/furniture_bench/data/low",
    repo_id: str = "hitsz-oyx/furniture_bench_low",
    fps: int = 10,
    output_base: str = None,
    num_workers: int = 4,
    parallel: bool = True,
):
    """
    将 Furniture Bench 数据集转换为 LeRobot 格式（优化版，支持并行）。
    
    参数:
        data_dir: 原始 .pkl 数据目录
        repo_id: HuggingFace 仓库 ID（同时作为输出目录名）
        fps: 数据采集帧率（用于生成时间戳）
        output_base: 输出基础目录，默认为 ~/.cache/huggingface/datasets/lerobot/
        num_workers: 并行工作进程数，默认为4（设置为0或1时使用顺序模式）
        parallel: 是否启用并行处理，默认为True
    """
    if output_base is None:
        output_base = Path.home() / ".cache" / "huggingface" / "datasets" / "lerobot"
    else:
        output_base = Path(output_base)
    
    output_path = output_base / repo_id.replace("/", "___")
    
    if output_path.exists():
        print(f"输出目录已存在，将覆盖旧文件: {output_path}")
    else:
        ensure_dir(output_path)
    print(f"输出目录: {output_path}")
    
    raw_path = Path(data_dir)
    pkl_files = list(raw_path.glob("**/*.pkl"))
    print(f"找到 {len(pkl_files)} 个 episode 文件")
    
    if not pkl_files:
        print(f"错误：在 {data_dir} 中没有找到 .pkl 文件")
        sys.exit(1)
    
    print("预计算总帧数...")
    total_frames = 0
    episode_frame_counts = []
    for pkl_file in tqdm(pkl_files, desc="扫描文件"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            count = len(data["observations"])
            episode_frame_counts.append(count)
            total_frames += count
    
    print(f"预计总帧数: {total_frames}")
    
    frame_indices = []
    current_idx = 0
    for pkl_file, count in zip(pkl_files, episode_frame_counts):
        frame_indices.append((pkl_file, fps, current_idx, output_path))
        current_idx += count
    
    if parallel and num_workers > 1:
        print(f"\n使用并行模式（{num_workers} 进程）...")
        total_written = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_episode, args) for args in frame_indices]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="转换 episodes"):
                total_written += future.result()
    
    else:
        print("\n使用顺序模式...")
        progress_bar = tqdm(total=total_frames, desc="转换并保存", unit="frame")
        total_written = 0
        
        for pkl_file, fps_val, start_frame_idx, out_path in frame_indices:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            
            observations = data["observations"]
            actions = np.array(data["actions"])
            
            for i, obs in enumerate(observations):
                img_main = obs["color_image1"]
                img_wrist = obs["color_image2"]
                
                if img_main.dtype != np.uint8:
                    img_main = (img_main * 255).astype(np.uint8)
                if img_wrist.dtype != np.uint8:
                    img_wrist = (img_wrist * 255).astype(np.uint8)
                
                rs = obs["robot_state"]
                pos = rs["ee_pos"]
                quat = rs["ee_quat"]
                gripper = rs["gripper_width"]
                
                state = np.concatenate([pos, quat, gripper]).astype(np.float32)
                
                act_idx = min(i, len(actions) - 1)
                action = actions[act_idx].astype(np.float32)
                
                timestamp = np.float32((start_frame_idx + i) / fps_val)
                
                frame_data = {
                    "image": img_main,
                    "cam_wrist": img_wrist,
                    "state": state,
                    "action": action,
                    "timestamp": timestamp,
                    "episode_index": np.int64(0),
                    "frame_index_in_episode": np.int64(i),
                }
                
                save_frame(out_path, start_frame_idx + i, frame_data)
                total_written += 1
                progress_bar.update(1)
        
        progress_bar.close()
    
    print("\n保存元数据...")
    metadata = {
        "num_frames": total_written,
        "fps": fps,
        "repo_id": repo_id,
        "features": {
            "image": {
                "dtype": "image",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channel"],
            },
            "cam_wrist": {
                "dtype": "image",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper_width"],
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": ["action"],
            },
        },
    }
    metadata_path = output_path / "metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n转换完成！")
    print(f"总帧数: {total_written}")
    print(f"输出目录: {output_path}")
    print(f"帧率: {fps} fps")


if __name__ == "__main__":
    tyro.cli(convert_furniture_bench_to_lerobot)