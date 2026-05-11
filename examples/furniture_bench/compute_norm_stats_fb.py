"""Compute normalization statistics for Furniture Bench dataset."""

import numpy as np
import tqdm
import tyro
from pathlib import Path
from safetensors import safe_open

import openpi.shared.normalize as normalize


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0
    
    def update(self, x):
        x = np.asarray(x)
        if x.ndim == 2:
            for row in x:
                self._update_single(row)
        else:
            self._update_single(x)
    
    def _update_single(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def get_statistics(self):
        if self.n < 2:
            return {"mean": self.mean, "std": np.ones_like(self.mean)}
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)
        return {"mean": self.mean, "std": std}


def main(
    data_dir: str = "/home/u2023312616/.cache/huggingface/datasets/lerobot/hitsz-oyx___furniture_bench_low",
    output_dir: str = "/home/u2023312616/test_ws/openpi/assets/hitsz-oyx/furniture_bench_low",
    max_frames: int | None = None,
):
    """
    Compute normalization statistics for Furniture Bench dataset.
    
    Args:
        data_dir: Directory containing the converted safetensors files
        output_dir: Directory to save the normalization statistics
        max_frames: Maximum number of frames to process (None for all)
    """
    data_path = Path(data_dir)
    
    safetensors_files = sorted(data_path.glob("frame_*.safetensors"))
    
    if max_frames is not None:
        safetensors_files = safetensors_files[:max_frames]
    
    print(f"Found {len(safetensors_files)} frames")
    
    stats = {
        "state": RunningStats(),
        "actions": RunningStats()
    }
    
    for filepath in tqdm.tqdm(safetensors_files, desc="Computing stats"):
        with safe_open(filepath, framework="np") as f:
            data = {k: f.get_tensor(k) for k in f.keys()}
        
        if "state" in data:
            stats["state"].update(data["state"])
        if "action" in data:
            stats["actions"].update(data["action"])
    
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    
    normalize.save(output_path, norm_stats)
    
    print("\nStatistics:")
    for key, stats in norm_stats.items():
        print(f"{key}:")
        print(f"  mean: {stats['mean']}")
        print(f"  std: {stats['std']}")


if __name__ == "__main__":
    tyro.cli(main)