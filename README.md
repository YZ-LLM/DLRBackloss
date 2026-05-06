# DLR-Backloss — Turkish LM (1 hour from scratch)

29M parameter Türkçe dil modeli, 2x T4 GPU üzerinde 1 saatte sıfırdan eğitilir.
Diagonal Linear Recurrent (DLR) blokları + Heinsen-style chunked parallel scan
+ u-normalize stabilitesi.

## Sonuçlar (1 saat, 2x T4)

| Metric | Değer |
|--------|-------|
| Params | 29.27M |
| Tokens seen | 203.5M (1 epoch) |
| Throughput | 56,500 tok/s (DDPx2) |
| Final loss | 4.12 |
| Final ppl | 61.6 |
| Steps | 6,211 |

## Mimari

- **DLRBlock**: `h_t = sigmoid(log_a) * h_{t-1} + u_t` (diagonal SSM)
- **scan_fp16**: Heinsen log-space chunked parallel scan (chunk=128)
  - Tek GPU sıralı scan: 5,843 tok/s
  - DDPx2 + chunked parallel: 56,500 tok/s (9.7x speedup)
- **u-normalize**: her chunk başında `s = max(|u|, 1)`, sonra `out *= s`.
  `exp(-log_p) * u` çarpımındaki fp32 overflow'u (t=128 için exp(87)≈1.6e37) önler.
- **Stability**: GRAD_CLIP=0.3, GradScaler init=2^10, growth=200, a clamp [0.5, 0.95]

## Kullanım

```bash
# 1) Veri hazırla (Türkçe corpus, 200M+ token önerilir)
python prepare_data.py

# 2) Eğit (Kaggle 2x T4)
torchrun --nproc_per_node=2 --master_port=29500 train_ddp.py

# 3) Generation
python generate.py "Türkiye'nin başkenti"
```

## Environment variables

- `TIME_LIMIT`: saniye, default 3600
- `LOG_EVERY`: step, default 100
- `DATA_PATH`: tokenized .npy
- `CKPT_PATH`: checkpoint kayıt yolu

## Kaggle ipucu

Uzun eğitimde Kaggle session restart riskine karşı:
- Her 1000 step'te checkpoint kaydet (kodu uyarla)
- `Save Version` ile output dahil kaydet
- Kernel'i idle bırakma, browser sekmesini kapat ama Save & Run All ile çalıştır
