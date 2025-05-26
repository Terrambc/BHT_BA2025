import torch
import torch.nn as nn


### Klassen ###

# 3.4.2 Implementierung einer kompakten SelfAttention Klasse
class SelfAttention_v1(nn.Module):
    # Inizialisierung der drei Gewichtsmatrizen
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # Werte von keys, value and queries mit der Matrixmultiplikation
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # Berechnung von den Attention Scores, Attention Weights, Context Vektor
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1 # Skalierung der Anttention Scores
        )

        context_vec = attn_weights @ values
        return context_vec


# Verbessserte SelfAttention Klasse
# Hier wird zur Berechnung der Gewichte die nn.Linear Funktion genutzt - Vorteil: nn.Linear 
# hat ein bevorzugtes Gewichtsinitialisierungsschema, das zu einem stabileren Modelltraining führt
class SelfAttention_v2(nn.Module):
    # Inizialisierung der drei Gewichtsmatrizen
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # Werte von keys, value and queries mit der Matrixmultiplikation
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Berechnung von den Attention Scores, Attention Weights, Context Vektor
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # Skalierung der Anttention Scores

        context_vec = attn_weights @ values
        return context_vec


# 3.5.3 Implementierung einer Kompakten Causal Self-Attention class
class CausalAttention(nn.Module):
    # Inizialisierung der drei Gewichtsmatrizen
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # Hinzufügen einer Dropout-Ebene

        # Puffer zusammen mit dem Modell werden automatisch auf das entsprechende Gerät (CPU oder GPU) verschoben. 
        # Damit wird sichergestellt, dass sich die Tensoren auf demselben Gerät wie die Modellparamter befinden. 
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # Tausch der Dimensionen 1 und 2, Batch-Dimension bleibt in der ersten Position (0)
        b, num_tokens, d_in = x.shape 

        # Für Inputs wo die "num_tokens" die "context_length" überschreitet würde es einen Fehler in the Maskenerzeugung weiter unten geben.
        # Werte von keys, value and queries mit der Matrixmultiplikation
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Berechnung von den Attention Scores, Attention Weights, Context Vektor

        # In PyTorch werden Operationen mit einem abschließenden Unterstrich direkt ausgeführt, wodurch unnötige Speicherkopien vermieden werden
        attn_scores.masked_fill_( 
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` für die Fälle, in denen die Anzahl der Token im Stapel kleiner ist als die unterstützte Kontextgröße
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        attn_weights = self.dropout(attn_weights) # Anwendung der Dropout Methode auf die Attention Weight Matrix

        context_vec = attn_weights @ values
        return context_vec


### 3.6.1 Erweiterung der Single-Head Attention zu einer Multi-head Attention ###
# Der Attention Mechanismus wird mehrfach ausgeführt
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)] # wie oft der Attention Mechanismus ausgeführt werden soll
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


### 3.6.2 Implementierung einer Multi-Head Attention mit Weight Splits ###
# Implementierung einer Stand-Alone MultiHeadAttention Klasse
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduziert die Projektionsdimension um sie an die gewünschte Ausgabendimension anzupassen

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Kombination von Head Outputs durch die Linear Ebene
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # aufteilung der Matix, indem eine Dimension 'num_heads' hinzugefügt wird
        # Unroll die letzte Dimension: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Umwandlung: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Berechnung des Scaled Dot-product Attention (aka Self-Attention) mit einer Causal Maske
        attn_scores = queries @ keys.transpose(2, 3)  # DSkalarprodukt für jeden Head

        # Ursprüngliche Maske auf die Anzahl der Token gekürzt und in Boolean konvertiert
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Verwenden Sie die Maske, um Attention Scores auszufüllen
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Kombiniere Heads, mit self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optionale Projektion

        return context_vec



### Funktionen ###

# Nutzung der Softmax-Funktion zur Normalisierung
# Einfache Implementierung der Softmax-Funktion
'''
Hinweis: Diese naive und sehr einfach Implementierung der Softmax-Funktion kann bei großen und kleinen Eingabewerten aufgrund von 
Überlauf- und Unterlaufproblemen zu numerischer Instabilität führen. 
'''
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)




def main():

    ### 3.3 Aufmerksamkeit auf die verschiedenen Teile der Eingabe von Self-Attention konzentrieren ###

    ### Schritt 1: Berechne unnormalisierte attention scores w ###
    # jede Zeile ist ein Wort und jede Spalte eine Embedding Dimension
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

    query = inputs[1]  # 2ter Input Token ist die Abfrage
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query) # Skalarprodukt (Transponierung hier nicht notwendig, da es sich um 1-dimensionale Vektoren handelt)

    print(attn_scores_2)

    ### 2. Schritt: Normalisierung der nicht normalisierten Attention Scores ("omegas"), sodass sie sich zu 1 summieren ###
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())


    # Nutzung der einfachen Softmax Funktion zur Normalisierung 
    attn_weights_2_naive = softmax_naive(attn_scores_2)

    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    # besser ist es die PyTorch Softmax Funktion zu nutzen, da die wesentlicher leistungsoptimiert ist
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())


    ### 3.Schritt: Berechnung des Kontextvektors ###
    # indem die embedded Input Token mit den Attention Weights multipliziert und
    # die erechneten Vektoren summiert werden

    query = inputs[1] # zweiter Input Token ist die Anfrage
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i

    print(context_vec_2)

    # Nun für alle Inputs Token
    attn_scores = torch.empty(6, 6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)

    print(attn_scores)

    # alternativ bekommt man dasselbe Ergebnis wenn man die Matrix Multiplikation nutzt
    attn_scores = inputs @ inputs.T
    print(attn_scores)

    # Normalisierung jeder Zeile, so das der Value der Zeile 1 ergibt
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    # Prüfung anhand einer Zeile, ob die Summe wirklich 1 ergibt
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)

    print("All row sums:", attn_weights.sum(dim=-1))

    # Berechnung aller Kontextvektoren
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    print("Previous 2nd context vector:", context_vec_2)


    ### 3.4 Implementierung des Self-Attention Mechnismus mit trainierbaren Weigts ###

    x_2 = inputs[1] # zweites Input Element
    d_in = inputs.shape[1] # Input Embedding Größe > d=3
    d_out = 2 # Die Output Größe > d=2

    torch.manual_seed(123)

    # Weight Matix > Query, Key, Value
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    # Berechnung der Vektoren
    query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
    key_2 = x_2 @ W_key 
    value_2 = x_2 @ W_value

    print(query_2)

    keys = inputs @ W_key 
    values = inputs @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    # Berechnung der nicht normalisierten Attention Scores > Skalarprodukt zwischen Query und Key Vektor
    keys_2 = keys[1] # Python starts index at 0
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)

    # für alle Attention Scores 
    attn_scores_2 = query_2 @ keys.T 
    print(attn_scores_2)

    
    # Berechnung der Attention Gewichte mit der Softmax Funktion plus Skalierung des Attention Scores: 
    # Dabei werden die Attention Scores durch die Quadratwurzel der Embedding Dimension dividiert. 
    d_k = keys.shape[1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print(attn_weights_2)

    # Berechung des Context Vektors (für Input Query Vektor 2)
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)

    # Aufruf der SelfAttention V1 Klasse
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print("\n\n")
    print(sa_v1(inputs))

    # Aufruf der SelfAttention V2 Klasse
    # Das Ergebnis ist unterschiedlich zu SelfAttention V1, da andere Anfangsgewichte genutzt werden
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print("\n")
    print(sa_v2(inputs))

    ### 3.5.1 Causal Attention Mechanismus mit Maskierung ###
    # Wiederverwendung der Query und Key Matrizen aus SelfAttention V2 Objekt
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs) 
    attn_scores = queries @ keys.T

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    print("\n")
    print(attn_weights)

    # zur Maskierung von Werten wird die PyTorch Tril Funktion genutzt, 
    # alles unterhalb der Matrixdiagonale wird auf 1 gesetzt, alles darüber auf 0
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("\n")
    print(mask_simple)
    
    # nun werden die Attention Weightes mit der Maske multiplizieren, un die Attentions Scores
    # über der Diagnole auf Null zu setzen. 
    masked_simple = attn_weights*mask_simple
    print("\n")
    print(masked_simple)

    # Normalisierung der Attention Weights
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print("\n")
    print(masked_simple_norm)


    # Verbesserung des Masking: Nicht normalisierte Attention Scores werden mit Negative Infinity maskiert, 
    # bevor die Softmatrix angewendet wird. 
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print("\n")
    print(masked)

    # Nach Anwendung der Softmax Funktion passen dann alle Werte zur Summe 1 pro Reihe. 
    # Es wird somit ein Arbeitsschritt gespart, was es effizienter macht
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
    print(attn_weights)


    ### 3.5.2 Masking zusätzliche Attention Weights mit Dropout ###
    # das reduziert Overfitting während des Trainings
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5) # dropout rate von 50%, die restlichen Werte werden mit 1/0.5 = 2 skaliert
    example = torch.ones(6, 6) # Erzeugt eine Matrix aus Einsen
    print(dropout(example))

    torch.manual_seed(123)
    print(dropout(attn_weights))

    
    # Anwendung der CausalAttention Klasse

    # Sicherstellung, dass die CausalAttention Klasse auch batch Eingaben handhaben kann
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) # 2 Inputs mit jeweils 6 Token und jedes Token hat eine Embedding Dimension von 3

    torch.manual_seed(123)

    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)

    context_vecs = ca(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)


    # Anwendung der Klasse MultiHeadAttentionWrapper
    torch.manual_seed(123)

    context_length = batch.shape[1] # Die Nummer der Token
    d_in, d_out = 3, 2 # bei d_out = zahl gibt an, wieviele Dimensionen dann herauskommen. z.B. d_out = 2 und num_heads = 2 ergibt 4 Dimensionen
    mha = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, 0.0, num_heads=2 # num_heads gibt vor wieviele Instanzen erzeugt werden
    )

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)



    # Anwendung der MultiHeadAttention Klasse

    torch.manual_seed(123)

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)



if __name__ == "__main__":
    main()

