"""
DLR-Backloss Turkish LM - DDP Training Script
- 2x T4 GPU, ~1 hour, 200M+ tokens, 29M params
- Heinsen-style chunked parallel scan with u-normalize
- u-normalize: prevents exp(-log_p) * u_c overflow in fp32
- GRAD_CLIP=0.3, scaler init=2^10, growth_interval=200
"""
import os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast


# ============================================================
# DDP setup
# ============================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


# ============================================================
# DLRBlock - Diagonal Linear Recurrent block
# ============================================================
class DLRBlock(nn.Module):
    def __init__(self, d, expand=2, layer_idx=0, n_layers=8):
        super().__init__()
        self.d = d
        self.H = d * expand
        self.in_proj = nn.Linear(d, self.H * 2)
        self.out_proj = nn.Linear(self.H, d)
        # log_a init: layer_idx'e göre farklı decay rates (multi-scale)
        # sigmoid(log_a) -> [0.5, 0.99]
        # log_a = log(p/(1-p)) (inverse sigmoid)
        # linspace(0.5, 0.99) ile başlat
        p = torch.linspace(0.5, 0.99, self.H)
        log_a_init = torch.log(p / (1 - p))  # logit(p)
        self.log_a = nn.Parameter(log_a_init)
        self.norm = nn.LayerNorm(d)
        # Zero-init out_proj — residual stabilitesi (GPT-2 / Mamba stili)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def scan_fp16(self, u, log_a, chunk=128):
        """
        Chunked parallel scan (Heinsen log-space).
        h_t = a*h_{t-1} + u_t, a=sigmoid(log_a) clamped to [0.5, 0.95]
        u-normalize: scale u per-chunk to prevent exp(-log_p)*u_c overflow.
        """
        B, T, H = u.shape
        a = torch.sigmoid(log_a).clamp(0.5, 0.95)  # >=0.5 chunked parallel stabilitesi için
        log_a_f = torch.log(a.float()).clamp(min=-0.69)  # log(0.5) = -0.69
        out = torch.zeros(B, T, H, device=u.device, dtype=torch.float32)
        h = torch.zeros(B, H, device=u.device, dtype=torch.float32)
        u_f = u.float()

        for c in range(0, T, chunk):
            ce = min(c + chunk, T)
            L = ce - c
            u_c = u_f[:, c:ce]
            # u-normalize: u_c'yi scale et, exp(-log_p)*u_c overflow'u önle
            s = u_c.abs().amax().clamp(min=1.0)
            u_c_n = u_c / s
            t_idx = torch.arange(L, device=u.device, dtype=torch.float32).view(1, L, 1)
            log_p = t_idx * log_a_f
            inv_u = torch.exp(-log_p) * u_c_n
            cum = torch.cumsum(inv_u, dim=1)
            sum_part = torch.exp(log_p) * cum * s  # geri scale
            decay_init = torch.exp((t_idx + 1) * log_a_f) * h.unsqueeze(1)
            out_c = sum_part + decay_init
            out[:, c:ce] = out_c
            h = out_c[:, -1]
        return out

    def forward(self, x):
        # x: (B, T, d)
        residual = x
        x = self.norm(x)
        gu = self.in_proj(x)  # (B, T, 2H)
        g, u = gu.chunk(2, dim=-1)
        g = F.silu(g)
        h = self.scan_fp16(u, self.log_a).to(u.dtype)
        y = g * h
        y = self.out_proj(y)
        return residual + y


# ============================================================
# DLR-Backloss F2 — full LM
# ============================================================
class DLRBacklossF2(nn.Module):
    def __init__(self, vocab=32000, d=512, n_layers=8, max_len=512):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([
            DLRBlock(d, expand=2, layer_idx=i, n_layers=n_layers)
            for i in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        # Tied embeddings
        self.lm_head.weight = self.tok_emb.weight
        # GPT-style init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not self.tok_emb.weight:  # head shared, skip
                    nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None and m.bias.requires_grad:
                    if not torch.allclose(m.bias, torch.zeros_like(m.bias)):
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Dataset
# ============================================================
class PackedDataset(Dataset):
    """Pre-tokenized .npy file, packed into seq_len blocks."""
    def __init__(self, npy_path, seq_len=512):
        self.ids = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n_blocks = (len(self.ids) - 1) // seq_len

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = torch.from_numpy(self.ids[i:i+self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.ids[i+1:i+self.seq_len+1].astype(np.int64))
        return x, y


# ============================================================
# Main training
# ============================================================
def main():
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)

    # --- config ---
    SEQ = 512
    B = 32  # per GPU
    LR = 3e-4
    GRAD_CLIP = 0.3  # düşük tutuyoruz, gradient explosion'a karşı
    TIME_LIMIT = int(os.environ.get("TIME_LIMIT", "3600"))
    LOG_EVERY = int(os.environ.get("LOG_EVERY", "100"))
    DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/working/data/combined_tr_32k_ids.npy")
    CKPT_PATH = os.environ.get("CKPT_PATH", "/kaggle/working/checkpoints/dlr_f2_v2.pt")

    if is_master:
        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- model ---
    model = DLRBacklossF2(vocab=32000, d=512, n_layers=8, max_len=SEQ).to(device)
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True, bucket_cap_mb=50)

    # --- data ---
    ds = PackedDataset(DATA_PATH, seq_len=SEQ)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=0,
                        pin_memory=True, drop_last=True)

    # --- optimizer ---
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler(init_scale=2**10, growth_interval=200)

    if is_master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[rank {rank}] Model: {n_params/1e6:.2f}M params, world_size={world_size}, LR={LR}", flush=True)
        print(f"[rank {rank}] Dataset: {len(ds)} blocks, B={B} per GPU, effective B={B*world_size}", flush=True)

    # --- warmup + constant LR ---
    WARMUP = 200
    def get_lr(step):
        if step < WARMUP:
            return LR * step / WARMUP
        return LR

    # --- train loop ---
    t0 = time.time()
    step = 0
    tokens_seen = 0
    epoch = 0
    while time.time() - t0 < TIME_LIMIT:
        sampler.set_epoch(epoch)
        for x, y in loader:
            if time.time() - t0 >= TIME_LIMIT:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            cur_lr = get_lr(step)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            step += 1
            tokens_seen += B * SEQ * world_size

            if step % LOG_EVERY == 0 or step == 1:
                loss_t = loss.detach().clone()
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                if is_master:
                    elapsed = time.time() - t0
                    tps = tokens_seen / elapsed
                    ppl = math.exp(min(loss_t.item(), 20))
                    print(f"step {step:5d} | lr {cur_lr:.2e} | tok {tokens_seen/1e6:6.1f}M | "
                          f"loss {loss_t.item():.4f} | ppl {ppl:8.1f} | gn {grad_norm:.2f} | "
                          f"tps {tps:,.0f} | t {elapsed:.0f}s", flush=True)
        epoch += 1

    # --- save ---
    if is_master:
        elapsed = time.time() - t0
        print(f"\n=== TRAIN DONE === elapsed={elapsed:.0f}s, tokens_seen={tokens_seen/1e6:.1f}M, "
              f"steps={step}, epochs={epoch}", flush=True)
        torch.save({
            "model": model.module.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {"vocab": 32000, "d": 512, "n_layers": 8, "max_len": SEQ},
            "tokens_seen": tokens_seen,
            "steps": step,
            "epochs": epoch,
        }, CKPT_PATH)
        print(f"[rank 0] Checkpoint saved: {CKPT_PATH}", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
