# Transformer De→En (from scratch)

Transformer encoder-decoder viết from scratch bằng PyTorch, huấn luyện dịch Đức → Anh trên dataset [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k).

Chuyển từ Colab notebook sang Python project chuẩn.

## Cấu trúc

```
transformer-project/
├── src/
│   ├── config.py       # tất cả hyperparameter
│   ├── evaluate.py     # đánh giá model bằng BLEU score (entry point)
│   ├── data.py         # tokenizer, vocabulary, dataset, dataloader
│   ├── model.py        # kiến trúc Transformer (attention, encoder, decoder...)
│   ├── train.py        # training loop (entry point)
│   ├── inference.py    # load checkpoint + dịch câu mới (entry point)
│   └── utils.py        # count_parameters, save/load checkpoint
├── checkpoints/         # model weights sau khi train
├── requirements.txt
└── .gitignore
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Nếu có GPU NVIDIA, cài bản torch có CUDA thay vì bản mặc định:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Train

```bash
python -m src.train
```

Model + vocab sẽ được lưu vào `checkpoints/transformer_de_en.pt` sau khi train xong.

Muốn đổi hyperparameter (d_model, epochs, batch_size...) thì sửa trực tiếp trong `src/config.py`.

## Dịch thử (inference)

Sau khi đã có checkpoint:

```bash
python -m src.inference
```

Mặc định file này test 3 câu tiếng Đức có sẵn. Muốn dịch câu khác, sửa `test_sentences` trong `src/inference.py`, hoặc import `translate()` từ code khác:

```python
from src.config import Config
from src.inference import load_model_for_inference, translate
from src.data import SimpleTokenizer
import torch

cfg = Config()
device = torch.device(cfg.device)
model, de_vocab, en_vocab = load_model_for_inference(cfg, device)
de_tokenizer = SimpleTokenizer("de")

result = translate(model, "ein hund läuft im park .", de_vocab, en_vocab, de_tokenizer, cfg, device)
print(result)
```
## Đánh giá model (BLEU)

Sau khi đã có checkpoint, chạy để đánh giá BLEU score trên tập test:

```bash
python -m src.evaluate
```

Script này dùng thư viện [`sacrebleu`](https://github.com/mjpost/sacrebleu) (implementation chuẩn, hay được dùng để so sánh kết quả giữa các paper/lab với nhau) và sẽ:

- Load checkpoint từ `checkpoints/transformer_de_en.pt`
- Dịch toàn bộ câu tiếng Đức trong test set bằng model đã train (greedy )
- Tính BLEU score bằng `sacrebleu.corpus_bleu(hypotheses, [references])` giữa bản dịch của model và câu tham chiếu tiếng Anh
- In ra BLEU score tổng thể, kèm vài ví dụ so sánh câu dịch model vs câu tham chiếu để kiểm tra định tính

