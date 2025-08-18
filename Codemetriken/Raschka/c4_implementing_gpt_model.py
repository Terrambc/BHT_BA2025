import torch
import torch.nn as nn
import tiktoken
from c3_coding_attention_mechanismus import MultiHeadAttention

'''
Kapitel 4 beschäftigt sich damit, die LLM Architektur zu erstellen. 
Eine GPT Architektur mit allen Subelementen. 
Dabei werden die bereits implementieren Klassen aus den ersten Kapiteln genutzt
und erweitert. 

1) GPT Placeholder model - um eine Übersicht der Struktur zu geben
2) Layer Normalization 
3) GELU Activation (Teil des Transformer Blocks)
4) Feed Forward Network (Teil des Transformer Blocks)
5) Shortcut Connections (Teil des Transformer Blocks)
6) Transformer Block (Teil des Transformer Blocks)
7) Finale GPT Architektur

'''



### Klassen ###


### 4.2 Implementierung der Normalisierungsklasse ###
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Durch den Wert eps wird ein Divisionsfehlder durch Null vermieden.
        # Er wird bei norm_x mit angehangen damit nicht 0/0 zum Fehler führt. 
        self.eps = 1e-5 # 0.001

        # zwei trainierbare Parameter scale und shift
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


### 4.3 Implementierung eines Feed Forward Networks mit GELU Aktivierung ###
# GELU ist eine glatte, nichtlineare Funktion, die ReLU annähert, jedoch mit einem von Null verschiedenen Gradienten für negative Werte
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

# Implementierung der Feed Forward Klasse
# ein kleines neurales Netzwerkmodul für den Transformer Block
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)




### 4.5 Verbinden von Attention und Linear Layer in einem Transformer Block ###
# Ein Transformatorblock kombiniert das kausale Multi-Head-Attention-Modul aus dem vorherigen Kapitel mit den 
# linearen Schichten, dem Feedforward-Neuronalen Netzwerk, das wir in einem früheren Abschnitt implementiert haben.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg) # FeedFoward Modul Aufruf
        self.norm1 = LayerNorm(cfg["emb_dim"]) # LayerNorm Modul Aufruf
        self.norm2 = LayerNorm(cfg["emb_dim"]) # LayerNorm Modul Aufruf
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut Connection für Attention Block
        # Aufbau wie in der Abbildung vom Transformer Block: LayerNorm1 -> Masked Multi-head Attention -> Dropout -> hinzufügen des Shortcuts (x)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # fügt den original Input wieder hinzu (Shortcut)

        # Shortcut Connection für  Feed Forward Block
        # Aufbau wie in der Abbildung vom Transformer Block: LayerNomr 2 -> Feed Forward -> Dropout -> hinzufügen des Shortcuts (x)
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # fügt den original Input wieder hinzu (Shortcut)

        return x


### 4.6 Erstellung des GPT Models ###
# Hinweis: Der Transformer Block wird beim kleinsten GPT Model (124M) 12 Mal wiederholt
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Aufruf des TransformerBlock Moduls
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Aufruf des LayerNorm Moduls
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


### 4.7 eine einfache Text-Generierungsfunktion für das GPT Model ###
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # #idx steht für ein Array von Indizes im aktuellen Kontext [batch, n_tokens]
    for _ in range(max_new_tokens):
        
        # Schneidet den Kontex ab, wenn die unterstützte Kontextgröße (context_size) überschritten wird
        # Beim kleinen Model wären es dann 1024 Tokens
        idx_cond = idx[:, -context_size:]
        
        # Holen der Vorhersage
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Konzentriert sich nur auf den letzten Linear Layer Output -> damit wird (batch, n_token, vocab_size) zu (batch, vocab_size)
        # Letzte Reihe 
        logits = logits[:, -1, :]  

        # Anwendung von Softmax Funktion um die Wahrscheinlichkeiten zu erhalten 
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Holen des idx vom Vokalbeleintrag mit dem höchsten Wahrscheinlichkeitswert
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Anhängen der gesammelten Indix an die laufenden Sequenz 
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx





### Tests und Ausführungen ###

def main():

    pass





if __name__ == "__main__":
    main()