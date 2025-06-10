import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from transformers import GPT2LMHeadModel




#------------------------------------------------------#
### Klassen ###

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

    
    def forward (self, x):
        B, T, C = x.size() # batch size, sequenze length, embedding dimensionality (n_embd)
        
        # Berechnet Query, Key, Values für alle Heads im Batch und schiebt den Head vorwärtzs zum nächsten Batch
        # nh ist "number of heads", hs ist "head size" und C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head = 12, hs = 64, so nh * hs = C = 768 Channels (Dimensionen bei Raschka) im Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        
        # Attention materialisiert die große Matrix (T, T) für alle Queries und Keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # Auffüllen mit dem "ignore_index"-Wert -inf
        att = F.softmax(att, dim=-1) # Normalisierung
        y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        

        ''' Funktioniert nicht auf CPU - nur auf CUDA '''
        # Dieser Codeteil ersetzt die darüber ausgeklammerten 4 Codezeilen
        # y = F.scale_dot_product_attention(q, k, v, is_causal=True)
        
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Alle Outputs Heads nebeneinander nach Größe wieder zusammenbauen

        # Output Projektion
        y = self.c_proj(y)
        
        return y




# Inizialisierung eines Multi-Level Perceptrons (MLPs) 
# MLP ist eine grundlegende neuronale Netzwerkarchitektur, die in 
# Transformer basierten Modellen angewendet wird 
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

@dataclass
# Konfigurationsdaten für das GPT Model
class GPTConfig: 
    block_size: int = 1024 # Maximale Squenz Länge -  maximale Anzahl von Input Tokens das Model händeln kann via dem Positional Embeddings
    vocab_size: int = 50257 # Anzahl der Token, bestehend aus 50.000 Byte-Pair-Embedded + 256 Bytes Tokens + 1 <|endoftext|> Sondertoken
    n_layer: int = 12 # Anzahl von Transformer Blöcken
    n_head: int = 12 # Anzahl der Attention Heads im Multi-head Attention Mechanismus
    n_embd: int = 768  # Größe des Embeddings - jeder Token wird in einen 768 dimensionalen Vektor umgewandelt


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

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # final Klassifikator 

        # die Gewichte von wte wird weitergleitet an das Element lm_head.weight
        self.transformer.wte.weight = self.lm_head.weight

        # Inizialisiert die Parameter
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx,targets=None):
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
        # Weiterleitung der final LayerNorm und den Klassifikator
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

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined vom model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M Parameter
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M Parameter
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M Parameter
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M Parameter
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
                # kopiere übere andere Parameter
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# DataLoader Version 1
class DataLoaderLiteV1:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # bei Inizialisierung Tokens von der Festplatte in den Speicher laden
        with open('BHT_BA2025_LLM_Karpathy/input.txt', 'r') as file:
            text = file.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # Status
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # Inputs
        y = (buf[1:]).view(B, T) # Targets

        # Die Positon des Tensors vorrücken
        self.current_position += B * T

        # wenn das Laden des nächsten Batch außerhalb der Grenzen liegt, einmal Reseten
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


'''
# Inizialisierung eines Data Loaders (Final Version)
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Hole die Shard Dateinamen
        data_root = "edu_fineweb10B"
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
        # die Position im Tensor vorrücken
        self.current_position += B * T * self.num_processes

        # Wenn das Laden des nächsten Batchs außerhalb der Grenzen liegt, fahre mit dem nächsten Shard fort
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
'''

### Tests und Ausführungen ###
def main():
    # Testen welches Gerät man nutzt cuda, mps oder cpu
    device = "cpu" # Fallcback - jeder hat eine CPU 
    if torch.cuda.is_available():
        device = "cuda" # für Nvidia User
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # für Apple Laptops
    print(f"using device: {device}")


    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # legt die interne Genauigkeit von Float32-Matrixmultiplikationen fest
    torch.set_float32_matmul_precision('high')

    # Logits
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    ''' model = torch.compile(model) <-- kann auf der CPU nur mit einer C++ Entwicklungsumgebung genutzt werden. Ist ansonsten für CUDA'''
    
    ### Beispiel 3: Dataloader und Trainingsdaten ###
    # Erstellung eines Trainingsdaten Loader
    train_loader = DataLoaderLiteV1(B=16, T=1024)    

    # Inizialisierung eines Optimizer für die Logits und Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(10):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # with torch.autocast(device_type=device, dtype=torch.bfloat16): <-- kann nicht auf der CPU genutzt werden, sorgt ansonsten für Freeze im System. 
        logits, loss = model(x, y) # Logits und Loss von Inputs und Targets

        loss.backward()
        optimizer.step()
        t1 = time.time()
        dt = (t1 - t0) * 1000 # Zeitspanne in Millisekunden
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f} ms, {tokens_per_sec:.2f}")



    import sys; sys.exit(0)
    # ---------------------------------------------------------------------------------- #
    ### Beispiel 2: Inputs, Targets, Loss - Vorbereitung für das Training vom Model ###
    # Ein Data Batch
    enc = tiktoken.get_encoding('gpt2')
    with open('BHT_BA2025_LLM_Karpathy/input.txt', 'r') as file:
        text = file.read()

    text = text[:1000] # die ersten 1000 Zeichen werden geladen
    tokens = enc.encode(text)
    B, T = 4, 32  # 4 Reihen a 32 TokensIDs
    buf = torch.tensor(tokens[:B*T + 1]) # Buffer 
    buf = buf.to(device)
    x = buf[:-1].view(B, T)  # Inputs
    y = buf[1:].view(B, T)   # Targets

    # Logits
    model = GPT(GPTConfig())
    model.to(device)
    # logits, loss = model(x, y) # Logits und Loss von Inputs und Targets

    # Inizialisierung eines Optimizer für die Logits und Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        optimizer.zero_grad()
        logits, loss = model(x, y) # Logits und Loss von Inputs und Targets
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")

   
    # ---------------------------------------------------------------------------------- #
    # Holen der Logits
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval() # das Model wird nicht trainiert
    model.to(device)
        
    ### Beispiel 1 ###
    # prefix Tokens
    
    num_return_sequences = 5
    max_length = 30

    enc = tiktoken.get_encoding('gpt2') # laden vom GPT2 Tokenizer
    tokens = enc.encode("Hello, I´m a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # Es wird folgendes generiert: x ist (B, T) davon B = 5, T = 8
    # setze den Seed auf 42
    torch.manual_seed(42)
    while x.size(1) < max_length:
        # Weiterleitung des Models um die Logits zu bekommen
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)

            # Nimmt die Logits von der letzten Position
            logits = logits[:, -1, :] # (B, vocab_size)

            # Holt die Wahrscheinlichkeiten
            probs = F.softmax(logits, dim=-1)

            # Gebe die 50 top-k Beispiele (huggingface pipeline default) wieder
            # topk_probs bekommt hier (5 ,50), topk_indicies sind (5, 50)
            topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)

            # Sammelt das Token von den top-k Wahrscheinlichkeiten
            ix = torch.multinomial(topk_probs, 1) # (B, 1)

            # sammelt die passenden Indizies
            xcol = torch.gather(topk_indicies, -1, ix) # (B, 1)

            # An die Sequenz anhängen
            x = torch.cat((x, xcol), dim=1)
    

    # Ausdruck des generierten Textes
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decode = enc.decode(tokens)
        print(">", decode)
    

  






if __name__ == "__main__":
    main()