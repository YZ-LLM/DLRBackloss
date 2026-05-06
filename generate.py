"""
Eğitilmiş checkpoint'ten Türkçe text generation.
Kullanım: python generate.py "Türkiye'nin başkenti"
"""
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from train_ddp import DLRBacklossF2


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100,
             temperature=0.8, top_k=40, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        if x.size(1) > 512:
            x_in = x[:, -512:]
        else:
            x_in = x
        logits = model(x_in)
        logits = logits[:, -1, :] / temperature
        # top-k
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        # EOS check
        if next_id.item() == tokenizer.token_to_id("[EOS]"):
            break
    out_ids = x[0].tolist()
    return tokenizer.decode(out_ids)


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = DLRBacklossF2(vocab=cfg["vocab"], d=cfg["d"],
                          n_layers=cfg["n_layers"], max_len=cfg["max_len"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"[OK] model yüklendi — {ck['tokens_seen']/1e6:.1f}M token, {ck['steps']} step")
    return model


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Türkiye'nin başkenti"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("/kaggle/working/checkpoints/dlr_f2_v2.pt", device)
    tokenizer = Tokenizer.from_file("/kaggle/working/data/tokenizer_tr.json")
    out = generate(model, tokenizer, prompt, max_new_tokens=100, device=device)
    print("=" * 60)
    print(out)
    print("=" * 60)
