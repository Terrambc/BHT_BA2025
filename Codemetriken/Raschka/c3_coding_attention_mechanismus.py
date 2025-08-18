import torch
import torch.nn as nn


### Klassen ###

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







def main():

    pass



if __name__ == "__main__":
    main()

