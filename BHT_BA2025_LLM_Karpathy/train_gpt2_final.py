import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))).view(1, 1, config.block_size, config.block_size )
        bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequenze length, embedding dimensionality (n_embd)

        # Berechnet Query, Key, Values für alle Heads im Batch und schiebt den Head vorwärtzs zum nächsten Batch
        # nh ist "number of heads", hs ist "head size" und C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head = 12, hs = 64, so nh * hs = C = 768 Channels (Dimensionen bei Raschka) im Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        ''' CUDA Funktioniert nicht auf CPU - nur auf CUDA
        # Dieser Codeteil ersetzt die unteren 4 Codezeilen
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash Attention
        '''
        # CPU
        # Attention materialisiert die große Matrix (T, T) für alle Queries und Keys
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
# MLP ist eine grundlegende neuronale Netzwerkarchitektur, die in 
# Transformer basierten Modellen angewendet wird 
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
    block_size: int = 1024 # Maximale Squenz Länge -  maximale Anzahl von Input Tokens das Model händeln kann via dem Positional Embeddings
    vocab_size: int = 50257 # Anzahl der Token, bestehend aus 50.000 Byte-Pair-Embedded + 256 Bytes Tokens + 1 <|endoftext|> Sondertoken
    n_layer: int = 12 # Anzahl von Transformer Blöcken (Layers)
    n_head: int = 12 # Anzahl der Attention Heads im Multi-head Attention Mechanismus
    n_embd: int = 768 # Größe des Embeddings - jeder Token wird in einen 768 dimensionalen Vektor umgewandelt


#-----------------------------------------------------------------------------
# Inizialisierung des GPT Models
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config


        # Nachbau des Hugging Face Transformersaufbau für GPT2
        # Erstellen des Main-Containers         
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weights Token Embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weights Position Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Hidden Layers, h = hidden
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
        # idx hat die Form (shape) (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Weiterleitung der Token- und Positionseinbettungen
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Form (T) - Positionindizies
        pos_emb = self.transformer.wpe(pos) # Positions Embedding der Form (T, n_embd)
        tok_emb = self.transformer.wte(idx) # Token Embeddings der Form (B, T, n_embd)
        x = tok_emb + pos_emb

        # Blöcke des Transformators weiterleiten
        for block in self.transformer.h:
            x = block(x)


        # Weiterleitung der fianllen LayerNorm und den Klassifikator
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # 3D Dimension werden in 2D Dimensionen umgewandelt für Inputs und Targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    # Ladet pre-trained GPT2-Model Gewichte von huggingface
    # Diese Methode gibt das GPT Objekt zurück, welches durch den Parameter model_type ausgewählt wurde.
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head und n_embd werden aus model_type bestimmt
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # immer 50257 für GPT-Modell-Checkpoints
        config_args['block_size'] = 1024 # immer 1024 für GPT-Modell-Checkpoints

        # erstellt ein von Grund auf initialisiertes minGPT-Modell
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # verwirft diese Maske / Puffer, keinen Parameter

        # initialisiere ein Huggingface/Transformers-Modell
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # kopiert, während sichergestellt ist, dass alle Parameter ausgerichtet sind und in Namen und Formen übereinstimmen
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignorier diese, nur ein Puffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # dasselbe, nur eine Maske (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Die OpenAI-Checkpoints verwenden grundsätzlich ein „Conv1D“-Modul, wir möchten aber nur ein Standard-Linear verwenden.
        # Das bedeutet, dass wir diese Gewichte beim Import transponieren müssen.
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # spezielle Behandlung für die Conv1D-Gewichte, die wir transponieren müssen
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # normales kopieren über andere Parameter
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # Konfiguration für den Optimize
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=False):

        # Startet mit allen Parametern die einen Grad erfordern
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Erstellt Optim Groups. Bei allen 2D-Parametern wird das Gewicht reduziert, andernfalls nicht.
        # d. h. alle Gewichtstensoren in Matmuls + Embeddings nehmen ab, alle Biases und Layernorms nicht.
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

        # Erstellt den AdamW-Optimierer und verwendet die Fused-Version, falls verfügbar
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

#----------------------------------------------------------------------------
# Laden der Tokens aus edufineweb Ordner
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

#-----------------------------------------------------------------------------
#  Inizialisierung eines Data Loaders (Final Version)
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Hole die Dateinamen der Shards
        data_root = "BHT_BA2025_LLM_Karpathy/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # Status, Initialisierung bei Shard Null
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # Inputs
        y = (buf[1:]).view(B, T) # Targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # Wenn das Laden des nächsten Batchs außerhalb der Grenzen liegt, fahre mit dem nächsten Shard fort
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


#--------------------------------------------------------------------------------
# Hilfsfunktion für die HellaSwag-Auswertung
# Erfasst Token, Maske und Logits und gibt den Index der Vervollständigung mit dem geringsten Verlust zurück.
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------


### Tests und Ausführungen ###
def main():
    # Ausführung der Trainingsschleife
    ''' Wieder eine Optimierung für CUDA '''
    # Das wird genutzt wenn man mehrere GDUs parallel laufen lässt. 
    # Setup DDP (Distributed Data Parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # Die Verwendung von DDP erfordert momentan CUDA, wir stellen das Gerät entsprechend dem Rang ein
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE']) # Anzahl der GPUs
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # Dieser Prozess übernimmt das Logging, Checkpointing etc. 
    else:
        # einfacher, non-DDP run
        # Wenn nur eine CPU oder eine GDU vorhanden ist
        # kein DDP Run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # Versuch automatisch das Gerät zu finden (Cuda oder CPU)
        # Testen welches Gerät man nutzt cuda, mps oder cpu
        device = "cpu" # Fallback - jeder hat eine CPU 
        if torch.cuda.is_available():
            device = "cuda" # für Nvidia User
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps" # für Apple Laptops
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    # Anpassung der Batchsize 
    '''total_batch_size = 524288 # 2**19, ~0.5M Anzahl von Tokens  <--- Freeze vom System. Vermutlich RAM Overflow'''
    total_batch_size = 8192 # läuft gerade noch mit meinem System 16384
    B = 8 # Micro Batch Size - orginal eingestellt 64
    T = 1024 # Länge der Sequenzen - original eingestellt 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0  # make sure total_batch_size ist Teilbar durch B * T
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Neuer Datensatz von Huggingface
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    # legt die interne Genauigkeit von Float32-Matrixmultiplikationen fest
    torch.set_float32_matmul_precision('high')

    # Erzeugt das Model - mit "hübschen Zahlen"
    model = GPT(GPTConfig(vocab_size=50304))

    # model = GPT.from_pretrained("gpt2") or init from OpenAI GPT-2
    ''' model = torch.compile(model) <-- kann auf der CPU nur mit einer C++ Entwicklungsumgebung genutzt werden. Ist ansonsten für CUDA'''
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model in den DDP Container
    raw_model = model.module if ddp else model # enthält immer das "raw" unwrapped model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

     # Funktion für die Lernrate
    def get_lr(it):
        # 1) lineares Aufwärmen für warmup_iters-Schritte
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) wenn it >  lr_decay_iters ist, gib die minimale Lernrate zurück
        if it > max_steps:
            return min_lr
        # 3) dazwischen Kosinusabfall bis zur minimalen Lernrate verwenden
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff startet mit 1 und geht dann gegen 0
        return min_lr + coeff * (max_lr - min_lr)

    # Inizialisierung eines Optimizer für die Logits und Loss
    # optimize!
    #  Erhöhung der Batchgröße schrittweise linear von einem kleinen Wert (32.000 Token) auf den vollen Wert über die ersten 4-12 Milliarden Token des Trainings
    # abhängig von der Modelgröße
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # Erstellt das Protokollverzeichnis, in das wir Prüfpunkte schreiben und protokollieren
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # Zum Schreiben öffnen, um die Datei zu löschen
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # bewertet ab und zu unseren Validierungsverlust
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    ''' kann nicht auf der CPU genutzt werden, sorgt ansonsten für Freeze im System. 
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)'''
                    
                    logits, loss = model(x, y) # Logits und Loss von Inputs und Targets für CPU
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                    
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optional Modell-Checkpoints schreiben
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    # Hinweis: Sie möchten möglicherweise auch optimizer.state_dict() hinzufügen und
                    # rng seeds usw., wenn Sie das Training genauer fortsetzen möchten
                    torch.save(checkpoint, checkpoint_path)

        # ab und zu hellaswag bewerten
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # nur Prozessbeispiele, bei denen i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue

                # Rendert das Beispiel in Token und Labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)

                # Holt sich die Logits
                with torch.no_grad():
                    '''kann nicht auf der CPU genutzt werden, sorgt ansonsten für Freeze im System. 
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)'''

                    logits, loss = model(tokens) # für CPU    
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            # Reduziert die Statistiken über alle Prozesse hinweg
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total

            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # ab und zu aus dem Modell generieren (außer Schritt 0, bei dem es sich um Rauschen handelt)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while xgen.size(1) < max_length:
                # Leitet das Modell weiter, um die Protokolle zu erhalten
                with torch.no_grad():

                    ''' kann nicht auf der CPU genutzt werden, sorgt ansonsten für Freeze im System. 
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size) '''

                    logits, loss = model(xgen) # (B, T, vocab_size) für CPU

                    # Nimmt die Logits an der letzten Position
                    logits = logits[:, -1, :] # (B, vocab_size)

                    # Holt sich die Wahrscheinlichkeiten
                    probs = F.softmax(logits, dim=-1)

                    # Gibt die 50 top-k Beispiele (huggingface pipeline default) wieder
                    # topk_probs bekommt hier (5 ,50), topk_indicies sind (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                    # Sammelt das Token von den top-k Wahrscheinlichkeiten
                    # Hinweis: Multinomial erfordert nicht, dass die Eingabe 1 ergibt
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)

                    # sammelt die passenden Indizies
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

                    # An die Sequenz anhängen
                    xgen = torch.cat((xgen, xcol), dim=1)

            # Ausdruck des generierten Textes
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # einen Schritt der Optimierung durchführen
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Feld wird für den Foward Pass verwendet
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            ''' kann nicht auf der CPU genutzt werden, sorgt ansonsten für Freeze im System.
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)'''
            
            logits, loss = model(x, y) # für CPU

            # Wir müssen den Verlust skalieren, um die Gradientenakkumulation zu berücksichtigen.
            # weil die Gradienten bei jedem nachfolgenden backwards() einfach addiert werden
            # Die Addition von Gradienten entspricht einer SUMME im Ziel, aber 
            # statt einer SUMME wollen wir MITTELWERT. Skalieren Sie den Verlust hier, damit er richtig ausfällt
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Beschneidung der Gradienten bei 1.0
        # Norm Clipping wird eingesetzt, um die Gradients nicht zu groß werden zu lassen 
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Bestimmt und setzt die Lernrate für diese Iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wartet auf die GPU, bis sie mit der Arbeit fertig ist

        # Zeitmessung    
        t1 = time.time()
        dt = t1 - t0 # Zeitspanne in Sekunden

        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()




if __name__ == "__main__":
    main()