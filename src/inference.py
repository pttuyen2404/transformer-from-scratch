"""
Inference: load checkpoint đã train và dịch câu mới.
Chạy: python -m src.inference
"""
import os

import torch

from src.config import Config
from src.data import SimpleTokenizer, Vocabulary
from src.model import Transformer


def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, cfg, device, max_len=50):
    """Dịch một câu mới."""
    tokens = src_tokenizer.tokenize(src_sentence)
    indices = [src_vocab.stoi.get(token, cfg.unk_idx) for token in tokens]
    indices = [cfg.bos_idx] + indices + [cfg.eos_idx]
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)
    encoder_output = model.encode(src_tensor, src_mask)

    tgt_indices = [cfg.bos_idx]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        decoder_output = model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
        output = model.fc_out(decoder_output)

        next_token = output.argmax(dim=-1)[:, -1].item()
        tgt_indices.append(next_token)

        if next_token == cfg.eos_idx:
            break

    tgt_tokens = [tgt_vocab.itos[idx] for idx in tgt_indices[1:-1]]
    return " ".join(tgt_tokens)


def _vocab_from_checkpoint(stoi, itos):
    """Dựng lại object Vocabulary từ dict đã lưu trong checkpoint."""
    vocab = Vocabulary()
    vocab.stoi = stoi
    vocab.itos = itos
    return vocab


def load_model_for_inference(cfg: Config, device):
    """Load model + vocab từ checkpoint đã train."""
    path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    checkpoint = torch.load(path, map_location=device)

    de_vocab = _vocab_from_checkpoint(checkpoint["de_stoi"], checkpoint["de_itos"])
    en_vocab = _vocab_from_checkpoint(checkpoint["en_stoi"], checkpoint["en_itos"])

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
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, de_vocab, en_vocab


def main():
    cfg = Config()
    device = torch.device(cfg.device)

    model, de_vocab, en_vocab = load_model_for_inference(cfg, device)
    de_tokenizer = SimpleTokenizer("de")

    test_sentences = [
        "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster .",
        "eine gruppe von männern lädt baumwolle auf einen lastwagen",
        "ein kleines mädchen klettert in ein spielhaus aus holz .",
    ]

    print("Testing translations:\n")
    for i, sentence in enumerate(test_sentences, 1):
        model.eval()
        translation = translate(model, sentence, de_vocab, en_vocab, de_tokenizer, cfg, device)
        print(f"{i}. German: {sentence}")
        print(f"   English: {translation}\n")


if __name__ == "__main__":
    main()
