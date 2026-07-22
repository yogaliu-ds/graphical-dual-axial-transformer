import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(price_path: str, high_path: str, low_path: str,
              open_path: Optional[str] = None, adjclose_path: Optional[str] = None,
              volume_path: Optional[str] = None) -> Tuple[np.ndarray, ...]:
    """Load and preprocess financial data with extended features

    Returns feature arrays in order:
    [simple return, range, close-open, volume_change_pct, adjclose_simple_return]
    """
    # Load basic data (existing)
    price_df = pd.read_csv(price_path, index_col=0)
    high_df = pd.read_csv(high_path, index_col=0)
    low_df = pd.read_csv(low_path, index_col=0)

    price_array = np.array(price_df)
    high_array = np.array(high_df)
    low_array = np.array(low_df)

    # Replace log returns with simple returns (percentage change)
    # Handle potential division by zero if previous price was 0
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = (price_array[1:, :] / price_array[:-1, :]) - 1
        range_returns = (high_array[1:, :] / low_array[1:, :]) - 1

    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    range_returns = np.nan_to_num(range_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    features = [returns, range_returns]

    # Feature 2: Close - Open (remains the same)
    if open_path is not None:
        open_df = pd.read_csv(open_path, index_col=0)
        open_array = np.array(open_df)
        close_open_diff = price_array[1:, :] - open_array[1:, :]
        features.append(close_open_diff)

    # Feature 3: Volume change percentage (remains the same)
    if volume_path is not None:
        volume_df = pd.read_csv(volume_path, index_col=0)
        volume_array = np.array(volume_df)
        # Handle division by zero and inf values
        volume_change_pct = np.zeros_like(volume_array[1:, :])
        valid_mask = (volume_array[:-1, :] > 0) & np.isfinite(volume_array[:-1, :]) & np.isfinite(volume_array[1:, :])
        volume_change_pct[valid_mask] = (volume_array[1:, :][valid_mask] - volume_array[:-1, :][valid_mask]) / volume_array[:-1, :][valid_mask]
        # Cap extreme values
        volume_change_pct = np.clip(volume_change_pct, -10, 10)
        features.append(volume_change_pct)

    # Feature 4: Simple return of adjClose
    if adjclose_path is not None:
        adjclose_df = pd.read_csv(adjclose_path, index_col=0)
        adjclose_array = np.array(adjclose_df)
        with np.errstate(divide='ignore', invalid='ignore'):
            adjclose_returns = (adjclose_array[1:, :] / adjclose_array[:-1, :]) - 1
        adjclose_returns = np.nan_to_num(adjclose_returns, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(adjclose_returns)

    return tuple(features)


def get_nearest_date_idx(date_index_str: np.ndarray, target_date: str) -> int:
    """Find the nearest date index for a target date"""
    idxs = np.where(date_index_str <= target_date)[0]
    if len(idxs) == 0:
        print(f"Warning: No data before {target_date}, using earliest date {date_index_str[0]}")
        return 0
    return idxs[-1]


def create_experiment_dirs(experiment_name: Optional[str] = None) -> dict:
    """Create single experiment directory with timestamp"""
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_dir = f"experiments/{experiment_name}"
    dirs = {
        'base': base_dir,
        'logs': base_dir,
        'models': base_dir, 
        'results': base_dir
    }
    
    os.makedirs(base_dir, exist_ok=True)
    
    return dirs


class ExperimentLogger:
    """Enhanced logger for experiments with structured output"""
    
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")
    
    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()
    
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.write("\n" + "="*50)
        self.write("\n EXPERIMENT CONFIGURATION")
        self.write("\n" + "="*50)
        self.write(f"\n{json.dumps(config, indent=2, ensure_ascii=False)}")
        self.write("\n" + "="*50 + "\n")
    
    def log_experiment_start(self, experiment_params: dict):
        """Log start of individual experiment"""
        self.write(f"\n{'='*20} EXPERIMENT START {'='*20}\n")
        for key, value in experiment_params.items():
            self.write(f"{key}: {value}\n")
        self.write(f"{'='*60}\n")
    
    def log_experiment_result(self, results: dict):
        """Log experiment results"""
        self.write(f"\n{'='*20} EXPERIMENT RESULTS {'='*20}\n")
        for metric, value in results.items():
            if isinstance(value, float):
                self.write(f"{metric:20}: {value:.4f}\n")
            else:
                self.write(f"{metric:20}: {value}\n")
        self.write(f"{'='*62}\n")


def load_adjacency_matrix(dataset_name: str) -> torch.Tensor:
    """Load adjacency matrix for a dataset"""
    adj_path = f"data/{dataset_name}/graphical_lasso/adj_matrix_{dataset_name}.npy"
    return torch.from_numpy(np.load(adj_path)).float()