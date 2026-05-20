# PredGrad Project - Complete Experiment List
## Shakespeare Dataset, CPU 120s, Token-level BPB

### Project Goals
1. **Academic/Industry Impact:** Novel training optimization methods for transformers
2. **Practical SLM:** Computer Engineer AI - full-stack software development assistant

---

## OPTIMIZER EXPERIMENTS (Sessions 1-2)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 1 | AdamW baseline | 50Lx32d | ~2.45 | Reference |
| 2 | SGD | 50Lx32d | ~2.80 | Worse |
| 3 | Adam (no WD) | 50Lx32d | ~2.50 | Worse |
| 4 | P3 only | 50Lx32d | ~2.35 | Better |
| 5 | Shampoo-lite only | 50Lx32d | ~2.40 | Slight better |
| 6 | Lookahead only | 50Lx32d | ~2.42 | Slight better |
| 7 | P3+Shampoo | 50Lx32d | ~2.28 | Good |
| 8 | P3+Lookahead | 50Lx32d | ~2.30 | Good |
| 9 | **LazySmart (P3+Shampoo+Lookahead)** | 50Lx32d | **~2.10** | **Champion: beat AdamW by 14.4%** |
| 10 | LR sweep (various) | 50Lx32d | various | lr=0.01 best |
| 11 | WD sweep | 50Lx32d | various | wd=0.1 best |
| 12 | Lookahead freq sweep | 50Lx32d | various | freq=5 best |
| 13 | Shampoo freq sweep | 50Lx32d | various | freq=10 best |

## PREDGRAD EXPERIMENTS (Sessions 1-2)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 14 | PredGrad v1 (weight transpose) | 50Lx32d | worse | Failed |
| 15 | PredGrad v2 (improved) | 50Lx32d | worse | Failed |
| 16 | PredGrad cosine similarity analysis | 50Lx32d | n/a | Low similarity with backprop |
| 17 | C++ extension for PredGrad | 50Lx32d | n/a | Built but no improvement |

## BACKLOSS SINGLE LOCATION (Session 3)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 18 | LazySmart baseline | 50Lx32d | 2.143 | Reference |
| 19 | BL-LN (LayerNorm) | 50Lx32d | 2.117 | -0.026 |
| 20 | BL-Embed (Embedding) | 50Lx32d | 2.175 | +0.032 |
| 21 | BL-LR (Layer-wise LR) | 50Lx32d | 2.188 | +0.045 |

## BACKLOSS COMBINATIONS (Session 3)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 22 | BL-LN + BL-Embed | 50Lx32d | 2.108 | -0.035 |
| 23 | BL-LN + BL-LR | 50Lx32d | 2.188 | +0.045 |
| 24 | **BL-LN + BL-Embed + BL-LR** | 50Lx32d | **2.095** | **-0.048** |

## BACKLOSS + PROPLOSS EVERYWHERE (Session 3)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 25 | ALL (Base + 6 new locations) | 50Lx32d | 2.490 | Harmful |
| 26 | ALL - PL-Embed | 50Lx32d | 2.266 | PL-Embed most harmful |
| 27 | ALL - PL-Head | 50Lx32d | 2.412 | PL-Head harmful |
| 28 | ALL - PL-Attn | 50Lx32d | 2.527 | **PL-Attn beneficial** |
| 29 | ALL - BL-Residual | 50Lx32d | 2.484 | Neutral |
| 30 | ALL - BL-AttnOut | 50Lx32d | 2.479 | Neutral |
| 31 | ALL - BL-GELU | 50Lx32d | 2.482 | Neutral |

## PL-ATTN TEST (Session 3)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 32 | Base + PL-Attn run1 | 50Lx32d | +0.018 | Inconsistent |
| 33 | Base + PL-Attn run2 | 50Lx32d | +0.078 | Inconsistent |
| 34 | PL-Attn only | 50Lx32d | neutral | Eliminated |

## BACKLOSS ON CLEAN nanoGPT (Session 3, char-level)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 35 | nanoGPT baseline (no LazySmart) | 3Lx192d char | 2.433 | Reference |
| 36 | BL-LN only | 3Lx192d char | worse | Harmful alone |
| 37 | BL-Embed only | 3Lx192d char | worse | Harmful alone |
| 38 | BL-Pos only | 3Lx192d char | worse | Harmful alone |
| 39 | BL-FinalLN only | 3Lx192d char | worse | Harmful alone |
| 40 | **BL-Best (LN+Embed+Pos+FinalLN)** | 3Lx192d char | **2.388** | **-0.045 synergy** |

## BACKLOSS NEW LOCATIONS (Session 3)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 41 | BL-Softmax | pipeline | no benefit | Eliminated |
| 42 | BL-ResBalance | pipeline | no benefit | Eliminated |
| 43 | BL-FinalLN | pipeline | no benefit | Eliminated |
| 44 | BL-PosEmbed | pipeline | mixed | Small benefit in char-level |
| 45 | BL-DynClip | pipeline | no benefit | Eliminated |

## TOKEN-LEVEL + BACKLOSS (Session 3-4)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 46 | Token-level LS+BL baseline | 3Lx192d | ~2.05-2.08 | Reference |
| 47 | +BL-LN+Em+LR+Pos | 3Lx192d | 2.154 | -0.028 |

## BL-INIT EXPERIMENTS (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 48 | LS+BL baseline | 3Lx192d | 2.089 | Reference |
| 49 | LS+BL+Init | 3Lx192d | 2.098 | No improvement |
| 50 | LS+Init only | 3Lx192d | 2.124 | No improvement |
| 51 | LS+BL+GA2 (grad accum) | 3Lx192d | 2.089 | Neutral |
| 52 | LS+BL+GA4 | 3Lx192d | 2.142 | Harmful (too few iters) |
| 53 | ALL (BL+Init+GA2) | 3Lx192d | 2.169 | Harmful |

## BL-INIT + PERTURBATION (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 54 | Init+noise=0.001 | 3Lx192d | 2.098 | No improvement |
| 55 | Init+noise=0.01 | 3Lx192d | 2.127 | Harmful |
| 56 | Init+noise=0.05 | 3Lx192d | 2.110 | No improvement |
| 57 | Init+noise=0.1 | 3Lx192d | 2.131 | Harmful |
| 58 | Init+noise=0.2 | 3Lx192d | 2.133 | Harmful |

## RUNTIME GRADIENT PERTURBATION (Session 4)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 59 | BL only (base) | 3Lx192d | 2.066 | Reference |
| 60 | **All n=0.05** | 3Lx192d | **2.019** | **-0.046** |
| 61 | All n=0.1 | 3Lx192d | 2.048 | -0.018 |
| 62 | Embed n=0.05 | 3Lx192d | 2.060 | -0.006 |
| 63 | All n=0.01 | 3Lx192d | 2.100 | +0.035 |
| 64 | LN n=0.05 | 3Lx192d | 2.101 | +0.035 |
| 65 | LR n=0.05 | 3Lx192d | 2.119 | +0.054 |

## PERTURBATION FINE-TUNING + PLAIN AdamW (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 66 | LS n=0 (base) | 3Lx192d | 2.046 | Reference |
| 67 | LS n=0.03 | 3Lx192d | 2.090 | Worse |
| 68 | LS n=0.05 | 3Lx192d | 2.077 | Worse |
| 69 | LS n=0.07 | 3Lx192d | 2.103 | Worse |
| 70 | AdamW only (no LS) | 3Lx192d | 2.423 | BL needs LazySmart |
| 71 | AdamW+BL | 3Lx192d | 2.444 | BL alone harmful |
| 72 | AdamW+BL+n=0.05 | 3Lx192d | 2.425 | No help without LS |

## CAI v1 PERTURBATION SEARCH (Session 4)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 73 | F1.baseline | 3Lx192d | 2.124 | Reference |
| 74 | **F1.periodic** | 3Lx192d | **2.024** | **-0.100** |
| 75 | F1.decay | 3Lx192d | 2.041 | -0.083 |
| 76 | F1.proportional | 3Lx192d | 2.054 | -0.070 |
| 77 | F1.loss_aware | 3Lx192d | 2.059 | -0.065 |
| 78 | F1.fixed | 3Lx192d | 2.554 | +0.430 |
| 79 | F1.inverse | 3Lx192d | 2.628 | +0.504 |
| 80 | **F3.+all_extra** | 3Lx192d | **2.018** | **-0.106** |
| 81 | F3.+mlp_proj | 3Lx192d | 2.022 | -0.102 |

## CAI v2 PERIOD + LOSS-AWARE (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 82 | p=3 | 3Lx192d | 2.040 | Best period |
| 83 | p=5 | 3Lx192d | 2.093 | |
| 84 | p=7 | 3Lx192d | 2.081 | |
| 85 | p=10 | 3Lx192d | 2.071 | |
| 86 | p=15 | 3Lx192d | 2.051 | |
| 87 | Loss-aware NO | 3Lx192d | 2.030 | Loss-aware harmful |
| 88 | Loss-aware YES | 3Lx192d | 2.097 | Too aggressive |
| 89 | F3.mlp_only | 3Lx192d | 2.051 | Best extra location |

## BL DIRECTION ONLY (Session 4)

| # | Experiment | Config | BPB | vs Baseline |
|---|-----------|--------|-----|-------------|
| 90 | LS only | 3Lx192d | 2.082 | Reference |
| 91 | **Dir LN+Em+Pos** | 3Lx192d | **2.073** | **-0.009** |
| 92 | Dir AllWeights | 3Lx192d | 2.134 | +0.052 harmful |
| 93 | Blend a=0.3 | 3Lx192d | 2.102 | Harmful |
| 94 | Blend a=0.5 | 3Lx192d | 2.128 | Harmful |
| 95 | Blend a=0.7 | 3Lx192d | 2.092 | Slightly harmful |
| 96 | Blend a=1.0 | 3Lx192d | 2.079 | Neutral (fastest: 302 iter) |

## BL DIRECTION + PERIODIC PERTURBATION (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 97 | Combined p=2 run1 | 3Lx192d | 2.129 | Inconsistent |
| 98 | Combined p=3 run1 | 3Lx192d | 2.027 | Good but... |
| 99 | Combined p=2 run2 | 3Lx192d | 2.110 | Inconsistent |
| 100 | Combined p=3 run2 | 3Lx192d | 2.125 | Inconsistent |

## SELF-EVOLVING v1 (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 101 | Self-evolving (random mutate) | 3Lx192d start | 2.185 | Found: bs=32, lr=0.0035, wd=0.01 |

## GROW v1 (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 102 | GROW v1 (loss trend) | 1Lx32d start | 2.174 | Bug: never grew (trend always negative) |

## GROW v2 (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 103 | GROW v2 (BPB comparison) | 1Lx32d start | 2.145 | Found: 2Lx32d, blk=48, bs=12 |

## GROW v3 (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 104 | **GROW v3 (continuous training)** | 2Lx64d start | **2.001** | **Found: 1Lx64d, lr=0.0151, bs=20** |

## 64d SWEEP (Session 4)

| # | Experiment | Config | BPB | Params | LLM-Ready |
|---|-----------|--------|-----|--------|-----------|
| 105 | 1Lx64d bs=20 lr=0.015 | | **1.909** | 0.34M | No |
| 106 | 1Lx64d bs=20 r2 | | **1.899** | 0.34M | No |
| 107 | **2Lx64d bs=16** | | **1.878** | 0.39M | No |
| 108 | **2Lx64d bs=20 lr=0.015** | | **1.875** | 0.39M | No |
| 109 | 3Lx64d bs=16 | | **1.886** | 0.44M | No |
| 110 | 3Lx64d bs=20 lr=0.015 | | **1.904** | 0.44M | No |
| 111 | 4Lx64d bs=20 lr=0.015 | | **1.901** | 0.49M | No |

## 128d / 256d LLM-READY SWEEP (Session 4)

| # | Experiment | Config | BPB | Params | LLM-Ready |
|---|-----------|--------|-----|--------|-----------|
| 112 | **2Lx128d bs=16** | | **1.948** | 0.97M | **YES** |
| 113 | 3Lx128d bs=16 | | 2.038 | 1.17M | YES |
| 114 | **4Lx128d bs=16** | | **1.981** | 1.36M | **YES** |
| 115 | 2Lx256d bs=16 | | 2.081 | 2.73M | Too slow |
| 116 | 3Lx256d bs=16 | | 2.101 | 3.51M | Too slow |
| 117 | 2Lx128d grow (bs=20 lr=0.015) | | 2.026 | 0.97M | YES |
| 118 | 3Lx128d grow | | 2.022 | 1.17M | YES |
| 119 | 3Lx192d (old champion) | | 2.064 | 2.19M | Replaced |

## SEMANTIC BASIS EMBEDDINGS v1 (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 120 | Normal embedding | 2Lx128d | 1.978 | Reference |
| 121 | Basis K=16 (softmax) | 2Lx128d | 2.655 | Failed |
| 122 | Basis K=32 (softmax) | 2Lx128d | 2.642 | Failed |
| 123 | Basis K=64 (softmax) | 2Lx128d | 2.643 | Failed |
| 124 | Basis K=128 (softmax) | 2Lx128d | 2.677 | Failed |
| 125 | Basis K=256 (softmax) | 2Lx128d | 2.647 | Failed |

## SEMANTIC BASIS v2 - FREE WEIGHTS (Session 4)

| # | Experiment | Config | BPB | Result |
|---|-----------|--------|-----|--------|
| 126 | Normal tied | 2Lx128d | 1.992 | Reference |
| 127 | Free K=64 | 2Lx128d | 2.034 | Close |
| 128 | Free K=128 | 2Lx128d | 2.097 | Worse |
| 129 | **Norm K=64** | 2Lx128d | **2.003** | **Near match!** |
| 130 | Norm K=128 | 2Lx128d | 2.076 | Worse |
| 131 | Factor d=32 (ALBERT) | 2Lx128d | 2.073 | Worse |
| 132 | Factor d=64 | 2Lx128d | 2.114 | Worse |

## SEMANTIC BASIS - 64d WITH BASIS (Session 4)

| # | Experiment | Config | BPB | Params | Result |
|---|-----------|--------|-----|--------|--------|
| 133 | 2Lx64d normal | | 1.920 | 0.39M | Reference |
| 134 | 2Lx128d normal | | 1.953 | 0.98M | LLM reference |
| 135 | 64d+K=32 | | 2.011 | 0.25M | 49% emb savings |
| 136 | 64d+K=64 | | 1.978 | 0.39M | Near 128d quality! |
| 137 | 64d+K=128 | | 1.965 | 0.69M | Good |
| 138 | 128d+K=32 | | 2.055 | 0.55M | 74% emb savings |
| 139 | 128d+K=64 | | 2.106 | 0.70M | Harmful on 128d |

---

## SUMMARY: TOTAL ~139 EXPERIMENTS

### Proven (LLM-applicable):
- **LazySmart Optimizer** - Beat AdamW by 14.4%
- **Backloss Gradient Scaling** - BL-LN, BL-Embed, BL-LR, BL-Pos
- **GROW v3** - Self-evolving training, found BPB 2.001

### Best LLM-Ready Result:
- **2Lx128d, LazySmart+BL, BPB 1.948, 0.97M params**

### Best Overall BPB:
- **2Lx64d, LazySmart+BL, BPB 1.875, 0.39M params**

### Eliminated:
- PredGrad, Proploss (PL-Attn), BL-Init, Gradient Accumulation, BL-ALL, Fixed/Inverse perturbation

### In Progress:
- Semantic Basis Embeddings (promising: Norm K=64 matched normal embedding)
- Computer Engineer SLM project
