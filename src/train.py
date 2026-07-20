"""
Entry point để train model. Chạy: python -m src.train
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import Config
from src.data import load_data, build_vocabularies, build_dataloaders
from src.model import Transformer
from src.utils import count_parameters, save_checkpoint

class NoamScheduler:
    """
    Learning rate scheduler theo công thức gốc paper "Attention Is All You Need".
    lr tăng tuyến tính trong warmup_steps bước đầu, sau đó giảm theo inverse sqrt(step).
    """
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self._last_lr = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr = lr
        self.optimizer.step()

    def _compute_lr(self):
        step = self.step_num
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_last_lr(self):
        return self._last_lr

def train_epoch(model, dataloader, scheduler, criterion, device, grad_clip):
    model.train()
    losses = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        scheduler.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scheduler.step()

        losses += loss.item()

    return losses / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            losses += loss.item()

    return losses / len(dataloader)


def main():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # --- Data ---
    train_data, val_data, _ = load_data(cfg)
    de_vocab, en_vocab, de_tokenizer, en_tokenizer = build_vocabularies(train_data, cfg)
    train_loader = build_dataloaders(
        train_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer, cfg, shuffle = True
    )
    val_loader = build_dataloaders(
        val_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer, cfg, shuffle = False
    )
    # --- Model ---
    model = Transformer(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    ).to(device)

    print(f"Model has {count_parameters(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    # --- Training loop ---
    print("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, scheduler, criterion, device, cfg.grad_clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Time: {end_time - start_time:.1f}s"
        )
    # Lưu checkpoint kèm vocab để inference.py dùng lại được
    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        cfg,
        scheduler.warmup_steps,
        extra={
            "de_stoi": de_vocab.stoi,
            "de_itos": de_vocab.itos,
            "en_stoi": en_vocab.stoi,
            "en_itos": en_vocab.itos,
        },
    )
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
