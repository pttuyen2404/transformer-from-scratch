"""
Tokenizer, Vocabulary, Dataset, DataLoader cho bài toán dịch Đức -> Anh.
Được tách ra từ 3 cell của notebook gốc (load dataset, build vocab, dataset/dataloader).
"""
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from src.config import Config


class SimpleTokenizer:
    """Tokenizer đơn giản - chỉ split by space và lowercase."""

    def __init__(self, language: str = "en"):
        self.language = language

    def tokenize(self, text: str):
        return text.lower().strip().split()


class Vocabulary:
    """Build vocabulary từ danh sách câu."""

    def __init__(self, freq_threshold: int = 2):
        self.itos = {0: "<unk>", 1: "<pad>", 2: "<bos>", 3: "<eos>"}
        self.stoi = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = 4  # sau các special token

        for sentence in sentence_list:
            tokens = tokenizer.tokenize(sentence)
            frequencies.update(tokens)

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text, tokenizer):
        tokens = tokenizer.tokenize(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.itos)


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]["de"]
        tgt_text = self.data[idx]["en"]

        src_indices = self.src_vocab.numericalize(src_text, self.src_tokenizer)
        tgt_indices = self.tgt_vocab.numericalize(tgt_text, self.tgt_tokenizer)

        src_indices = [self.src_vocab.stoi["<bos>"]] + src_indices + [self.src_vocab.stoi["<eos>"]]
        tgt_indices = [self.tgt_vocab.stoi["<bos>"]] + tgt_indices + [self.tgt_vocab.stoi["<eos>"]]

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


class MyCollate:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]

        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx)

        return src_batch, tgt_batch


def load_data(cfg: Config):
    """Tải dataset Multi30k từ HuggingFace."""
    print("Loading Multi30k dataset from HuggingFace...")
    dataset = load_dataset(cfg.dataset_name)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, val_data, test_data


def build_vocabularies(train_data, cfg: Config):
    """Build vocab tiếng Đức và tiếng Anh từ tập train."""
    print("Building vocabularies...")
    de_tokenizer = SimpleTokenizer("de")
    en_tokenizer = SimpleTokenizer("en")

    de_vocab = Vocabulary(freq_threshold=cfg.freq_threshold)
    en_vocab = Vocabulary(freq_threshold=cfg.freq_threshold)

    de_sentences = [item["de"] for item in train_data]
    en_sentences = [item["en"] for item in train_data]

    de_vocab.build_vocabulary(de_sentences, de_tokenizer)
    en_vocab.build_vocabulary(en_sentences, en_tokenizer)

    print(f"German vocab size: {len(de_vocab)}")
    print(f"English vocab size: {len(en_vocab)}")

    return de_vocab, en_vocab, de_tokenizer, en_tokenizer


def build_dataloaders(train_data, val_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer, cfg: Config):
    """Tạo DataLoader cho train/val."""
    train_dataset = TranslationDataset(train_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer)
    val_dataset = TranslationDataset(val_data, de_vocab, en_vocab, de_tokenizer, en_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=MyCollate(cfg.pad_idx),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        collate_fn=MyCollate(cfg.pad_idx),
    )

    print("Data loaders created!")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    return train_loader, val_loader
