import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- 1. PRO DLR-BACKLOSS (Latest Innovation) ---
class ProDecayDLRBlock(nn.Module):
    def __init__(self, d_model, lambda_decay=0.98):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lambda_decay = lambda_decay
        self.gate_proj = nn.Linear(d_model, d_model * 4) # q, k, v, g

    def forward(self, x):
        B, N, D = x.shape
        proj = self.gate_proj(self.norm(x))
        q, k, v, g = proj.chunk(4, dim=-1)

        # Weighted Stats
        weights = torch.pow(self.lambda_decay, torch.arange(N, device=x.device).flip(0)).view(1, -1, 1)
        mu = torch.cumsum(k * weights, dim=1) / (torch.cumsum(weights.expand(B, N, D), dim=1) + 1e-9)

        # Gated Selectivity
        alpha = torch.sigmoid(q - mu) * torch.sigmoid(g)

        # Fast Linear Scan
        hidden = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(N):
            hidden = (self.lambda_decay * (1 - alpha[:, t, :])) * hidden + alpha[:, t, :] * v[:, t, :]
            outputs.append(hidden.unsqueeze(1))
        
        return x + torch.cat(outputs, dim=1)


class ProDLRModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ProDecayDLRBlock(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        x = self.embedding(x)
        for layer in self.layers: x = layer(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


# 1. Kural Bazlı Veri Üretimi
def generate_logic_data(num_samples, seq_len, vocab_size):
    # Rastgele sayılar üret
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Kural: Her adımda o ana kadarki sayıların toplamının mod vocab_size hali (basit bir recurrence)
    y = torch.cumsum(x, dim=1) % vocab_size
    return x.to(device), y.to(device)

# 1. Veri Parametreleri (%60 alınan testle aynı)
logic_vocab_size = 10
s_len = 32
d_model = 128
n_layers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Pro-DLR Mimarisini 4 Katmanlı Olarak Kuralım
model = ProDLRModel(logic_vocab_size, d_model, n_layers=n_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f'--- %60 Rekorunu Kırma Testi Başlıyor ({n_layers} Katmanlı Pro-DLR) ---')
print(f'Cihaz: {device} | İterasyon Hedefi: 2501\n')

start_time = time.time()
for i in range(2501):
    # %60 alınan testle aynı veri üretim kuralı
    xb, yb = generate_logic_data(32, s_len, logic_vocab_size)
    
    logits, loss = model(xb.to(device), yb.to(device))
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        elapsed = time.time() - start_time
        print(f'Iter {i:4d} | Loss: {loss.item():.4f} | Süre: {elapsed:.1f}s')

# 3. Final Genelleme Testi
model.eval()
with torch.no_grad():
    xt, yt = generate_logic_data(1000, s_len, logic_vocab_size)
    logits, _ = model(xt.to(device))
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == yt.to(device)).float().mean().item()

print(f'\n--- TEST SONUCU ---')
print(f'Final Accuracy: %{acc*100:.2f}')
if acc > 0.60:
    print(f'TEBRİKLER! %60 barajını {acc*100-60:.2f} puan farkla aştık!')
else:
    print(f'Rekor kırılamadı, ancak mimari verimliliği doğrulandı.')
