# v9 — GERCEK kelime testi (tokenizer artifact'ini ele)
import torch, math, copy, numpy as np
import torch.nn.functional as F

WORK = "/kaggle/working"
DEVICE = torch.device("cuda")

# m, stats, Wf, base_state v8.5 hücresinden hâlâ bellek'te
# Kontrol:
print("m, stats, Wf, base_state bellek'te:", "m" in dir(), "stats" in dir(), "Wf" in dir(), "base_state" in dir())

from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{WORK}/tokenizer/tok_tr_16k.json")
VOCAB = tok.get_vocab_size()

# Vocab'daki gerçek kelimeleri bul: en az 4 karakter, alfabetik
vocab_dict = tok.get_vocab()
real_words = []
for token_str, tid in vocab_dict.items():
    decoded = tok.decode([tid])
    s = decoded.strip()
    if len(s) >= 4 and s.isalpha() and s.isascii() == False:  # Türkçe karakter içeren
        real_words.append((tid, decoded, s))
    elif len(s) >= 5 and s.isalpha():
        real_words.append((tid, decoded, s))

print(f"  Gercek-kelime kandidat: {len(real_words)}")

# Bu kandidatlar arasından train+val'da geçen, frekansa göre 3 grup
ids_arr = np.load(f"{WORK}/data/wiki_tr_ids.npy")
train_ids = ids_arr[:-200_000]
val_ids = ids_arr[-200_000:]
val_counts = torch.from_numpy(np.bincount(val_ids, minlength=VOCAB))
train_cnt = stats["cnt"].cpu()

# Filter: train>=20 ve val>=10
candidates = [(tid, dec) for tid, dec, _ in real_words 
              if train_cnt[tid] >= 20 and val_counts[tid] >= 10]
candidates.sort(key=lambda x: -train_cnt[x[0]].item())
print(f"  Filtreli (train>=20 & val>=10): {len(candidates)}")

# 3 grup, 3 token
n = len(candidates)
selected = {
    "sik":   candidates[:3],
    "orta":  candidates[n//2:n//2+3],
    "nadir": candidates[-3:] if n > 6 else candidates[3:6],
}
print("\n  Secilen gercek kelimeler:")
for grp, items in selected.items():
    for tid, dec in items:
        print(f"    {grp:5s} id={tid:5d} '{dec}'  train={int(train_cnt[tid]):4d}  val={int(val_counts[tid]):4d}")

# Test (var olan eval_token_val ve fonksiyonları kullan)
# Yoksa redefine:
SEQ, BATCH = 128, 8
val_t = torch.from_numpy(val_ids).long()
def get_batch(arr, bs=BATCH, sl=SEQ, seed=None):
    g = torch.Generator(); g.manual_seed(seed) if seed else None
    ix = torch.randint(0, len(arr)-sl-1, (bs,), generator=g if seed else None)
    x = torch.stack([arr[i:i+sl] for i in ix])
    y = torch.stack([arr[i+1:i+1+sl] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def eval_token_val(model, target_token, n_batches=100):
    model.eval()
    correct = 0; total = 0
    for b in range(n_batches):
        x, y = get_batch(val_t, seed=44000+b)
        mask_t = (y == target_token)
        nt = mask_t.sum().item()
        if nt == 0: continue
        logits = model(x)
        log_t = logits[mask_t]; y_t = y[mask_t]
        correct += (log_t.argmax(-1) == y_t).sum().item(); total += nt
    return (correct/total if total else float('nan'), total)

target_norm = m.lm_head.weight.data.norm(dim=-1).mean().item()

print(f"\n  {'grp':<7} {'tid':<6} {'kelime':<25} {'full':<8} {'rand':<8} {'form':<8} {'val_n':<6}")
results_v9 = {}
for grp, items in selected.items():
    for tid, dec in items:
        m.load_state_dict(base_state)
        a_full, n_full = eval_token_val(m, tid)

        m.load_state_dict(base_state)
        with torch.no_grad():
            torch.manual_seed(1000+tid)
            d = m.lm_head.weight.size(1)
            m.lm_head.weight.data[tid] = torch.randn(d, device=DEVICE) * (target_norm / math.sqrt(d))
            m.tok_emb.weight.data[tid] = torch.randn(d, device=DEVICE) * 0.02
        a_rand, _ = eval_token_val(m, tid)

        m.load_state_dict(base_state)
        with torch.no_grad():
            m.lm_head.weight.data[tid] = Wf[tid]
            m.tok_emb.weight.data[tid] = stats["c_mean"][tid]
        a_form, _ = eval_token_val(m, tid)

        results_v9[(grp, tid)] = {"full": a_full, "rand": a_rand, "form": a_form, "n": n_full}
        print(f"  {grp:<7} {tid:<6} {dec[:24]:<25} {a_full:<8.3f} {a_rand:<8.3f} {a_form:<8.3f} {n_full:<6}")

# OZET
print("\n  GRUP OZET:")
for grp in ["sik", "orta", "nadir"]:
    rs = [r for k, r in results_v9.items() if k[0] == grp]
    if not rs: continue
    full = np.mean([r["full"] for r in rs])
    rand = np.mean([r["rand"] for r in rs])
    form = np.mean([r["form"] for r in rs])
    print(f"  {grp}: full={full:.3f}  rand={rand:.3f}  form={form:.3f}  ratio={form/max(full,1e-6)*100:.0f}%")

torch.save(results_v9, f"{WORK}/v9_real_words.pt")
print("\n=== TAMAMLANDI ===")
print("Yorum:")
print("- Eger gercek kelimelerde de form > full -> bulgu GERCEK")
print("- Eger gercek kelimelerde form ~ rand -> v8 sonuclari TOKENIZER ARTIFACT")
print("- Eger form > full ama %120 gibi kucuk fark -> mekanizma var ama mutevazi")
