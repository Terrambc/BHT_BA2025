import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ------------------------------------------------------
### Modellklassen ###

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

# ------------------------------------------------------
### Daten-Loader ###

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        # data_root = "BHT_BA2025_LLM_Karpathy/edu_fineweb10B"
        data_root = "BHT_BA2025_LLM_Karpathy/tinystory_npy_vergleichbar"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

# ------------------------------------------------------
### MLM-Evaluation / Next-Token-Prediction ###

def evaluate_mlm(model, dataloader, device, num_batches=10):
    """
    Führt eine Next Token Prediction- (MLM-) Evaluation auf dem übergebenen Modell und Dataloader durch.
    Gibt Loss und Accuracy zurück.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    for _ in range(num_batches):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        shift_logits = logits[:, :-1, :]
        shift_labels = y[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='sum'
        )
        predictions = shift_logits.argmax(-1)
        correct += (predictions == shift_labels).sum().item()
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    avg_loss = total_loss / total_tokens
    accuracy = correct / total_tokens
    print(f"MLM-Evaluation (Next Token Prediction): Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    return avg_loss, accuracy

# ------------------------------------------------------
### Trainingsschleife inkl. MLM-Evaluation ###

def main():
    device = torch.device("cpu")
    print("Using device:", device)
    torch.manual_seed(1337)

    # Modell- und Trainingsparameter (klein für CPU!)
    config = GPTConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    total_batch_size = 1024
    B = 4
    T = 256
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader = DataLoaderLite(B=B, T=T, split="val")

    model = GPT(config)
    model.to(device)

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 20
    max_steps = 1000

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Validierungsverlust
        if step % 10 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 2
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # MLM-Evaluation (Next Token Prediction)
        if step % 20 == 0 or last_step:
            print("Starte MLM-Evaluierung...")
            avg_loss, accuracy = evaluate_mlm(model, val_loader, device, num_batches=5)
            with open(log_file, "a") as f:
                f.write(f"{step} mlm_eval loss={avg_loss:.4f} acc={accuracy:.4f}\n")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt

        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if __name__ == "__main__":
    main()
