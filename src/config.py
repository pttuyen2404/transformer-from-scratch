"""
Cấu hình tập trung cho toàn bộ project.
Thay vì hardcode rải rác như trong notebook, mọi hyperparameter nằm ở đây.
"""
from dataclasses import dataclass
import torch


@dataclass
class Config:
    # --- Dataset ---
    dataset_name: str = "bentrevett/multi30k"
    freq_threshold: int = 2          # tần suất tối thiểu để 1 từ được đưa vào vocab

    # --- Model architecture ---
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 5000              # cho positional encoding

    # --- Training ---
    batch_size: int = 64
    lr: float = 1e-4
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    epochs: int = 30
    grad_clip: float = 1.0

    # --- Special tokens (cố định theo Vocabulary trong data.py) ---
    unk_idx: int = 0
    pad_idx: int = 1
    bos_idx: int = 2
    eos_idx: int = 3

    # --- Runtime ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "transformer_de_en.pt"
