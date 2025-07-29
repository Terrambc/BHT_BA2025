'''
Modifizierte Version von Karpathys Code mit PyTorch DataLoader
für Performance-Vergleich in der Bachelorarbeit
'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import psutil
import random
from transformers import GPT2LMHeadModel

#------------------------------------------------------#
### Klassen und Funktionen  ###

# Implementierung der Causal Self Attention Klasse
# Multi-Head Attention 
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value Vorhersagen für alle Heads in einem Batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Vorhersage für den Output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Regularisierung > einer Gruppe von Methoden beim maschinellen Lernen zur Vermeidung von Überanpassung
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

    def forward(self, x):
        B, T, C = x.size() # Bedeutung der Abkürzungen B = batch size, T = sequenze length, C = embedding dimensionality (n_embd)

        # Berechnet Query, Key, Values für alle Heads im Batch und schiebt den Head vorwärts zum nächsten Batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # CPU - Attention materialisiert die große Matrix (T, T) für alle Queries und Keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # Auffüllen mit dem "ignore_index"-Wert -inf
        att = F.softmax(att, dim=-1) # Normalisierung
        y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Alle Outputs Heads nebeneinander nach Größe wieder zusammenbauen

        # Output Vorhersage
        y = self.c_proj(y)
        return y

#-----------------------------------------------------------------------------
# Inizialisierung eines Multi-Level Perceptrons (MLPs) 
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#-----------------------------------------------------------------------------
# Inizialisierung des Blocks in der GPT Klasse
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

#-----------------------------------------------------------------------------
# Konfigurationsdaten für das GPT Model
@dataclass
class GPTConfig:
    block_size: int = 1024  # reduziert auf 512
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

#-----------------------------------------------------------------------------
# Inizialisierung des GPT Models
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Nachbau des Hugging Face Transformersaufbau für GPT2
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weights Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weights Position Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Hidden Layers
            ln_f = nn.LayerNorm(config.n_embd), # Final Layer Norm
        ))

        # final Klassifikator 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # die Gewichte von wte wird weitergleitet an das Element lm_head.weight
        self.transformer.wte.weight = self.lm_head.weight

        # Inizialisiert die Parameter
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
        
        # Weiterleitung der Token- und Positionseinbettungen
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # Blöcke des Transformators weiterleiten
        for block in self.transformer.h:
            x = block(x)

        # Weiterleitung der finalen LayerNorm und den Klassifikator
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Berechne Loss falls Targets vorhanden
        loss = None
        if targets is not None:
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=False):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device_type
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

#----------------------------------------------------------------------------
# PyTorch Dataset für NumPy Shards
class NumpyDataset(Dataset):
    def __init__(self, shard_paths, T, master_process=False):
        self.sequences = []
        
        if master_process:
            print(f"Loading {len(shard_paths)} shards into memory...")
        
        for shard_path in shard_paths:
            # Lade NumPy Shard
            npt = np.load(shard_path)
            npt = npt.astype(np.int32)
            tokens = torch.tensor(npt, dtype=torch.long)
            
            # Erstelle Sequenzen der Länge T+1 für Input/Target Paare
            for i in range(0, len(tokens) - T, T):
                sequence = tokens[i:i + T + 1]
                if len(sequence) == T + 1:
                    self.sequences.append(sequence)
        
        if master_process:
            print(f"Created {len(self.sequences)} sequences of length {T+1}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def pytorch_collate_fn(batch, device="cpu"):
    """Collate function für PyTorch DataLoader"""
    # Stack alle Sequenzen
    batch_tensor = torch.stack(batch)
    
    # Erstelle Input/Target Paare
    inputs = batch_tensor[:, :-1]   # Alle außer letztem Token
    targets = batch_tensor[:, 1:]   # Alle außer erstem Token
    
    return inputs.to(device), targets.to(device)

#----------------------------------------------------------------------------
# Hilfsfunktionen
def load_tokens(filename):
    """Original Karpathy load_tokens Funktion für Kompatibilität"""
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def create_pytorch_dataloaders(data_root, B, T, device, master_process=False):
    """Erstelle PyTorch DataLoaders aus NumPy Shards"""
    
    # Sammle Shard-Pfade
    all_files = os.listdir(data_root)
    train_shards = [os.path.join(data_root, f) for f in all_files if 'train' in f and f.endswith('.npy')]
    val_shards = [os.path.join(data_root, f) for f in all_files if 'val' in f and f.endswith('.npy')]
    
    train_shards = sorted(train_shards)
    val_shards = sorted(val_shards)
    
    if master_process:
        print(f"Found {len(train_shards)} train shards and {len(val_shards)} val shards")
    
    # Erstelle Datasets
    train_dataset = NumpyDataset(train_shards, T, master_process)
    val_dataset = NumpyDataset(val_shards, T, master_process)
    
    # Erstelle DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: pytorch_collate_fn(batch, device),
        num_workers=0,  # Für CPU-Training
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=B,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: pytorch_collate_fn(batch, device),
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def calculate_accuracy(logits, targets):
    """Berechnet die Next-Token-Prediction Accuracy"""
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    mask = targets != -1
    accuracy = (correct * mask).sum() / mask.sum()
    return accuracy.item()

def safe_next_batch(data_loader, data_iter_ref):
    """Sichere next_batch Funktion die automatisch den Iterator zurücksetzt"""
    try:
        return next(data_iter_ref[0])
    except (StopIteration, TypeError):
        data_iter_ref[0] = iter(data_loader)
        return next(data_iter_ref[0])

### Tests und Ausführungen ###
def main():
    # Setup DDP (gleich wie Original)
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Seed setup
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    enc = tiktoken.get_encoding('gpt2')

    # Batch-Konfiguration
    total_batch_size = 4096
    B = 4
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # NEUE PyTorch DataLoaders erstellen
    data_root = "BHT_BA2025_LLM_Karpathy/tinystory_npy_vergleichbar"
    train_loader, val_loader = create_pytorch_dataloaders(data_root, B, T, device, master_process)
    
    # Berechne Gesamtanzahl der Tokens wie bei Raschka
    if master_process:
        total_train_tokens = 0
        for batch in train_loader:
            inputs, _ = batch
            total_train_tokens += inputs.numel()
        
        total_val_tokens = 0
        for batch in val_loader:
            inputs, _ = batch
            total_val_tokens += inputs.numel()
        
        print(f"PYTORCH Train tokens: {total_train_tokens:,}")
        print(f"PYTORCH Val tokens: {total_val_tokens:,}")
        print(f"GESAMT: {total_train_tokens + total_val_tokens:,}")

    torch.set_float32_matmul_precision('high')

    # Model setup (gleich wie Original)
    model = GPT.from_pretrained('gpt2')
    model.to(device)
    
    use_compile = False
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Training Parameter
    max_lr = 5e-5
    min_lr = max_lr * 0.1
    warmup_steps = 800
    max_steps = 8498

    # WandB Setup
    if master_process:
        wandb.init(
            project="Analyse_Raschka_Karpathy",
            name="karpathy_pytorch_dataloader_Seed123",
            config={
                "batch_size": B,
                "total_batch_size": total_batch_size,
                "sequence_length": T,
                "learning_rate": max_lr,
                "min_learning_rate": min_lr,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "weight_decay": 0.1,
                "grad_accum_steps": grad_accum_steps,
                "model_config": raw_model.config.__dict__,
                "device": device,
                "num_parameters": sum(p.numel() for p in raw_model.parameters()),
                "dataloader_type": "pytorch"  # Wichtig für Vergleich!
            }
        )

        wandb.define_metric("step")
        wandb.define_metric("train_loss", step_metric="step")
        wandb.define_metric("val_loss", step_metric="step")
        wandb.define_metric("train_perplexity", step_metric="step")
        wandb.define_metric("val_perplexity", step_metric="step")
        wandb.define_metric("train_accuracy", step_metric="step")
        wandb.define_metric("val_accuracy", step_metric="step")
        wandb.define_metric("throughput", step_metric="step")
        wandb.define_metric("memory_usage", step_metric="step")

    # Learning Rate Function
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

    # Logging Setup
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_pytorch.txt")
    with open(log_file, "w") as f:
        pass

    # Iterator-Referenzen für DataLoader
    train_iter_ref = [None]
    val_iter_ref = [None]

    # Training Loop
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Evaluation
        if step % 400 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 5
                val_accuracy_accum = 0.0

                for _ in range(val_loss_steps):
                    x, y = safe_next_batch(val_loader, val_iter_ref)
                    
                    logits, loss = model(x, y)
                    accuracy = calculate_accuracy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                    val_accuracy_accum += accuracy / val_loss_steps
                    
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_accuracy_accum, op=dist.ReduceOp.AVG)

            if master_process:                   
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                val_perplexity = torch.exp(val_loss_accum).item()   

                wandb.log({
                    "val_loss": val_loss_accum.item(),
                    "val_perplexity": val_perplexity,
                    "val_accuracy": val_accuracy_accum,
                    "step": step
                })

                if step > 0 and (step % 8498 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_pytorch_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)

        # Text Generation
        if ((step > 0 and step % 850 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 50
            tokens = enc.encode("Once upon a time,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(123 + ddp_rank)

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)

            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

            if last_step:
                print("\n" + "="*50)
                print("FINAL GENERATED TEXT (Step {})".format(step))
                print("="*50)
                final_tokens = enc.encode("Once upon a time,")
                final_tokens = torch.tensor(final_tokens, dtype=torch.long)
                final_tokens = final_tokens.unsqueeze(0)
                xgen_final = final_tokens.to(device)
                
                max_length_final = 100
                while xgen_final.size(1) < max_length_final:
                    with torch.no_grad():
                        logits, _ = model(xgen_final)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                        xcol = torch.gather(topk_indices, -1, ix)
                        xgen_final = torch.cat((xgen_final, xcol), dim=1)
                
                final_text = enc.decode(xgen_final[0].tolist())
                print(f"\nFinal story: {final_text}")
                print("="*50 + "\n")

        # Training Step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = safe_next_batch(train_loader, train_iter_ref)

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        t_after_backward = time.time()
        if step % 100 == 0:
            print(f"Step {step}: Data loading + forward/backward: {t_after_backward - t0:.2f}s")

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.no_grad():
            train_accuracy = calculate_accuracy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0

        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

            train_perplexity = torch.exp(loss_accum).item()

            if device_type == "cuda":
                memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024 
            else:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024 

            cpu_percent = psutil.cpu_percent(interval=None)
            memory_usage_percent = psutil.virtual_memory().percent
            
            # Wandb logging
            if step % 400 == 0 or last_step:
                wandb.log({
                    "train_loss": loss_accum.item(),
                    "train_perplexity": train_perplexity,
                    "train_accuracy": train_accuracy, 
                    "learning_rate": lr,
                    "grad_norm": norm,
                    "throughput": tokens_per_sec,
                    "memory_usage": memory_usage_mb,
                    "time_per_step_ms": dt * 1000, 
                    "step": step,
                    "system/cpu_percent": cpu_percent,
                    "system/memory_percent": memory_usage_percent
                })

            # Progress Output
            if step % 400 == 0 or last_step:
                print(f"Step {step:4d} | "
                    f"train_loss {loss_accum.item():.4f} | "
                    f"train_perplexity {train_perplexity:.1f} | "
                    f"train_acc {train_accuracy:.3f} | "
                    f"tok/s {tokens_per_sec:.1f} | mem {memory_usage_mb:.0f} MB")

    if master_process:
        wandb.finish()

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()