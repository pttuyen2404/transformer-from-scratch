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


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip):
    model.train()
    losses = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

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
    train_data, val_data, test_data = load_data(cfg)
    de_vocab, en_vocab, de_tokenizer, en_tokenizer = build_vocabularies(train_data, cfg)
    train_loader, val_loader = build_dataloaders(
        train_data, val_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer, cfg
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

    # --- Training loop ---
    print("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Val Loss: {val_loss:.3f} | "
            f"Time: {end_time - start_time:.1f}s"
        )

    print("\nTraining complete!")

    # Lưu checkpoint kèm vocab để inference.py dùng lại được
    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        cfg,
        extra={
            "de_stoi": de_vocab.stoi,
            "de_itos": de_vocab.itos,
            "en_stoi": en_vocab.stoi,
            "en_itos": en_vocab.itos,
        },
    )


if __name__ == "__main__":
    main()
