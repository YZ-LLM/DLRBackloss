
# BACKLOSS FORMÜL — Türkçe Açıklama

## Temel mantık

LLM bir token tahmin ederken iki adımlı çalışıyor:

1. **Hidden state (h)** üretir: girdi cümlesini d=192 boyutunda bir vektöre sıkıştırır. Bu vektör "şu an ne hakkında konuşuluyor"u temsil eder.
2. **lm_head** linear katmanı: bu hidden state'i alır, vocab boyutunda (16000) skor üretir. Hangi token gelecek?

`logit[v] = h · W_lm[v]` (iç çarpım). En yüksek logit'li token kazanır.

**Eğitim sırasında:** Model her token v için W_lm[v] satırını gradient descent ile yavaş yavaş öğrenir. Nadir token'lar az gradient aldığı için kötü öğrenilir.

## Formül ne yapıyor?

Eğitim yerine, W_lm[v] satırını **kapalı formla** doğrudan hesaplıyoruz:

```
W_lm[v] = (μ_v − μ_global) / σ_global
```

Burada:
- **μ_v** = "Token v'nin geldiği pozisyonlardaki hidden state'lerin ortalaması"
  - Yani train corpus'unu tarıyoruz, `next_token = v` olan her pozisyonda hidden state'i topluyoruz, ortalama alıyoruz
  - Bu vektör "v gelmek üzereyken hidden state nasıl görünür"ün portresi
- **μ_global** = Tüm pozisyonlarda hidden state ortalaması (genel arka plan)
- **σ_global** = Tüm pozisyonlarda hidden state standart sapması (boyut başı)

`(μ_v - μ_global) / σ_global` bize **"v token'ı genel ortalamadan hangi yönde sapıyor?"**ü veriyor — z-score gibi.

## Recency-weighted detay

Hidden state'leri toplarken son pozisyonu kullanmak yerine, **EMA (exponential moving average)** uyguluyoruz:

```
h_eff(t) = α · h_eff(t-1) + (1-α) · h(t)    α=0.85
```

Sebep: bir token'ın "mood"u birden fazla pozisyonun karışımıdır, sadece anlık değil. Bu small detail formülü daha sağlam yapıyor.

## Embedding tarafı

Aynı formül `tok_emb[v]` (giriş embedding) için de kullanılıyor:

```
tok_emb[v] = μ_v   (z-score normalize değil, ham mean)
```

Yani v token'ı görüldüğünde, modele "bu hidden state-ish bir şey gör" diyen embedding olarak μ_v'yi atıyoruz.

## Kullanım — 3 senaryo

### Senaryo 1: Yeni token eklemek (vocab+1)
"Modelimde 'kuantum' kelimesi yoktu, şimdi eklemek istiyorum"
1. 'kuantum' kelimesini vocab'a ekle (id=16000)
2. Bu kelimenin geçtiği bir corpus topla (10 cümle yeter)
3. O cümleleri modelden hidden state çıkar, μ_kuantum hesapla
4. W_lm[16000] = (μ_kuantum - μ_global) / σ_global
5. tok_emb[16000] = μ_kuantum
6. **Eğitim YOK.** Model artık 'kuantum'u tanıyor.

### Senaryo 2: Mevcut token'ı yenilemek
Model 'Ankara'yı kötü öğrenmiş (acc 0.000). Düzeltmek için:
1. Train corpus'tan 'Ankara'nın geçtiği pozisyonları topla
2. μ_Ankara hesapla
3. W_lm[ankara_id] satırını formülle güncelle
4. **Saniyeler içinde** 'Ankara' acc 0.000 → 0.875

### Senaryo 3: 1 dakika fine-tune ile mükemmel
Senaryo 2 sonrası, sadece o token satırını **freeze fine-tune** ile cilala:
1. lm_head + tok_emb hariç tüm parametreleri freeze
2. Sadece o token'ın satırlarına gradient ulaşmasına izin ver (mask)
3. 600 step train (1 dakika)
4. Acc 0.875 → **1.000**

Bizim deneyde nadir token id=18 ('millî') tam böyle: form_noft 0.968 → form_ft **1.000**.

## Neden çalışıyor?

Eğitim, gradient descent ile her satırı yavaş yavaş "iyi yöne" iter. Çok sample gören token'larda iyi sonuç verir, az sample gören token'larda zayıf kalır. **Formül o yönü tek-shot hesaplar** — corpus'taki istatistiği direkt kullanır. Bu yüzden nadir token'larda eğitilmiş modelden ÇOK daha iyi çalışıyor.

Cosine(W_form, W_lm_eğitilmiş) = 0.147 düşük çünkü ikisi farklı yönler. Ama acc her ikisi de yüksek çünkü hidden state geometrisinde **birden fazla doğru yön** var (overdetermined system).



-------------------------------
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
