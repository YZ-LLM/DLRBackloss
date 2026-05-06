"""
Tokenizer eğitimi (BPE 32K) + corpus tokenize → packed .npy
HuggingFace tokenizers kullanır (sentencepiece olmadan).
"""
import os
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Sequence


def train_tokenizer(text_files, vocab_size=32000, save_path="tokenizer_tr.json"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC()])  # lowercase YOK — Türkçe için kötü (İ/i ayrımı)
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        min_frequency=2,
    )
    tokenizer.train(text_files, trainer)
    tokenizer.save(save_path)
    print(f"[OK] tokenizer kaydedildi: {save_path}")
    return tokenizer


def tokenize_corpus(text_files, tokenizer_path, output_npy, dtype=np.int32):
    tok = Tokenizer.from_file(tokenizer_path)
    all_ids = []
    for fpath in text_files:
        print(f"tokenize: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tok.encode(line).ids
                all_ids.extend(ids)
                all_ids.append(tok.token_to_id("[EOS]"))
    arr = np.array(all_ids, dtype=dtype)
    np.save(output_npy, arr)
    print(f"[OK] {len(arr)/1e6:.1f}M token kaydedildi: {output_npy}")
    return arr


if __name__ == "__main__":
    # Türkçe corpus dosyaları (örnek: oscar-tr, wikipedia-tr, vb.)
    TXT_FILES = [
        "/kaggle/input/turkish-corpus/oscar_tr.txt",
        "/kaggle/input/turkish-corpus/wiki_tr.txt",
    ]
    os.makedirs("/kaggle/working/data", exist_ok=True)

    # 1) Tokenizer eğit (~5-10 dk)
    train_tokenizer(TXT_FILES, vocab_size=32000,
                     save_path="/kaggle/working/data/tokenizer_tr.json")

    # 2) Corpus'u tokenize et (~10-15 dk)
    tokenize_corpus(TXT_FILES,
                    tokenizer_path="/kaggle/working/data/tokenizer_tr.json",
                    output_npy="/kaggle/working/data/combined_tr_32k_ids.npy")
