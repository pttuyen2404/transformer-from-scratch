'''
Evaluation: tính toán các metric như BLEU, ROUGE để đánh giá chất lượng dịch.
Chạy: python -m src.evaluate
'''
import sacrebleu
import torch
from src.config import Config
from src.inference import load_model_for_inference,translate
from data.data import load_data,SimpleTokenizer

def evaluate_bleu(model, test_data, de_vocab, en_vocab, de_tokenizer, cfg, device):
    """
    test_data: list các cặp (de_sentence, en_sentence) - câu thô, chưa tokenize.
    """
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for de_sentence, en_reference in zip(test_data['de'], test_data['en']):

            hyp = translate(
                model, de_sentence, de_vocab, en_vocab, de_tokenizer, cfg, device
            )
            hypotheses.append(hyp)
            references.append(en_reference)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references

def main():
    cfg = Config()
    device = torch.device(cfg.device)

    # Load model checkpoint
    model,de_vocab,en_vocab = load_model_for_inference(cfg, device)

    # Load evaluation data
    _, _, test_data = load_data(cfg)
    print(f"Số câu test: {len(test_data)}")
    de_tokenizer = SimpleTokenizer(language="de")
    bleu_score, hypotheses, references = evaluate_bleu(model, test_data, de_vocab, en_vocab, de_tokenizer, cfg, device)
    print(f"BLEU score: {bleu_score:.2f}")
    print(f"Số câu test: {len(test_data)}\n")

    print("Một vài ví dụ:")
    for h, r in zip(hypotheses[:5], references[:5]):
        print(f"HYP: {h}")
        print(f"REF: {r}")
        print("---")

if __name__ == "__main__":
    main()