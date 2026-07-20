"""Helper functions dùng chung."""
import os
import torch


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, cfg, warmup_steps, extra=None):
    """Lưu checkpoint model + optimizer + vocab size, để load lại dùng inference sau này."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "warmup_steps": warmup_steps,
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, path, optimizer=None, map_location=None):
    """Load checkpoint. Trả về dict chứa metadata (vd. epoch, vocab)."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded: {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint
