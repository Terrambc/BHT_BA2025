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

# Implementierung von Dummy Klassen 
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Platzhalter für TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Platzhalter für LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Einfacher Platzhalter

    def forward(self, x):
        # Dieser Block tut nichts und gibt nur ihre Eingabe zurück.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # Die Parameter hier dienen lediglich dazu, die LayerNorm-Schnittstelle nachzuahmen.

    def forward(self, x):
        # Diese Schicht tut nichts und gibt nur ihre Eingabe zurück.
        return x

### 4.2 Implementierung der Normalisierungsklasse ###
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Durch den Wert eps wird ein Divisionsfehlder durch Null vermieden. 
        self.eps = 1e-5

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




### 4.4 Hinzufügen von Shortcut Verbindungen
# Eine Abkürzungsverbindung schafft einen alternativen, kürzeren Pfad für den Gradientenfluss durch das Netzwerk
# Dies wird erreicht, indem die Ausgabe einer Schicht zur Ausgabe einer späteren Schicht hinzugefügt wird, 
# wobei normalerweise eine oder mehrere Schichten dazwischen übersprungen werden.
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Berechnet den Output des aktuellen Layers
            layer_output = layer(x)
            # Überprüfung ob Shortcut genutzt werden kann 
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x



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
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut Connection für Attention Block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # fügt den original Input wieder hinzu

        # Shortcut Connection für  Feed Forward Block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # fügt den original Input wieder hinzu

        return x


### 4.6 Erstellung des GPT Models ###
# Hinweis: Der Transformer Block wird beim kleinsten GPT Model (124M) 12 Mal wiederholt

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
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







### Funktionen ###

# Gradienten ausdrucken
def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Berechnen des Verlusts basierend darauf, wie nahe das Target und Output sind
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass zur Berechnung der Gradienten
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
             # Druckt den mittleren absoluter Gradienten der Gewichte
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")



### 4.7 eine einfache Text-Generierungsfunktion für das GPT Model ###
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # #idx steht für ein Array von Indizes im aktuellen Kontext [batch, n_tokens]
    for _ in range(max_new_tokens):
        
        # Schneidet den Kontex ab, wenn die unterstützte Kontextgröße (context_size) überschritten wird
        idx_cond = idx[:, -context_size:]
        
        # Holen der Vorhersage
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Konzentriert sich nur auf den letzten time step -> damit wird (batch, n_token, vocab_size) zu (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Anwendung von Softmax Funktion um die Wahrscheinlichkeiten zu erhalten to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Holen des idx vom Vokalbeleintrag mit dem höchsten Wahrscheinlichkeitswert
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Anhängen der gesammelten Indix an die laufenden Sequenz 
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx





### Tests und Ausführungen ###

def main():

    ### Variablen ###
    # GPT Model mit 124 Millionen Parameter
    GPT_CONFIG_124M = {

        "vocab_size": 50257,    #  # Vokabelgröße - in dem Fall 50527 Wörter
        "context_length": 1024, # maximale Anzahl von Input Tokens das Model kann händeln via dem Positional Embeddings
        "emb_dim": 768,         # Größe des Embeddings - jeder Token wird in einen 768 dimensionalen Vektor umgewandelt
        "n_heads": 12,          # Anzahl der Attention Heads im Multi-head Attention Mechanismus
        "n_layers": 12,         # Anzahl von Transformer Blöcken
        "drop_rate": 0.1,       # Dropout Rate um Overfitting entgegenzuwirken - in dem Fall 10%
        "qkv_bias": False       # Bestimmte ob ein Bias-Vektor in den linearen Schichten des Multi-head Attention Mechanismus für Query-Key-Value Berchnungen mit einbezogen werden soll. 
    }



    # Ausführung des DummyGPT Models
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
   

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

    ### 4.2 Normalisieren von Aktivierungen mit Layer-Normalisierung ###
    torch.manual_seed(123)

    # Erstellt 2 Trainingsbeispiele mit jeweils 5 Dimensionen (Features)
    batch_example = torch.randn(2, 5) 

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)


    # Mittelwert
    mean = out.mean(dim=-1, keepdim=True)
    # Varianz
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    
    # Durch das Subtrahieren des Mittelwerts und Dividieren durch die Quadratwurzel
    # der Varianz werden die Inputs so zentriert, dass sie übe4r die gesamte 
    # Spaltendimension (Feature) hinweg einen Mittelwert von 0 und eine Varianz von 1 aufweisen.
    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:\n", out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)

    # Ausführung der Klasse LayerNorm 
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    
    # Ausführung der FeedForward Klasse
    ffn = FeedForward(GPT_CONFIG_124M)

    # input shape: [batch_size, num_token, emb_size]
    x = torch.rand(2, 3, 768) 
    out = ffn(x)
    print(out.shape)

    ### 4.4 Shortcut Klasse testen ###
    layer_sizes = [3, 3, 3, 3, 3, 1]  

    sample_input = torch.tensor([[1., 0., -1.]])

    torch.manual_seed(123)

    # ohne Shortcut
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)

    # mit Shortcut
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print("\n")
    print_gradients(model_with_shortcut, sample_input)

    ### 4.5 Ausführung der TransformerBlock Klasse ###
    torch.manual_seed(123)

    x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


    ### 4.6 Nutzung der neuen GPTModel Klasse ###
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    # für die Analyse der Größe dieser Modelarchitektur
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    ### 4.7 Testen der Textgenerierung ###
    print("\n\n")
    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() # disable dropout

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output:", out)
    print("Output length:", len(out[0]))
    
    # Batchdimension entfernen und zurück in Text konvertieren
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)





if __name__ == "__main__":
    main()