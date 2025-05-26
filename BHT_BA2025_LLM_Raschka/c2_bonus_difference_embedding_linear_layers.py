import torch



# drei Traingsbeispiele die Token-IDs im LLM-Kontext darstellen
idx = torch.tensor([2,3,1])

'''
Die Anzahl der Zeilen in der Embedding Matrix lässt sich durch die höchste Token-ID + 1 bestimmen. 
Wenn die höchste Token-ID 3 ist, benötigen wir 4 Zeilen für die möglichen Token-IDs 0, 1, 2 und 3.
'''

num_idx = max(idx)+1

# der gewünschte Embedding Dimensionsparameter ist ein Hyperparameter
'''
Hyperparameter sind im Machine Learning externe, konfigurationsabhängige Variablen, die verwendet werden
um den Traininsgprozess von Modellen zu steuern. 
'''
out_dim = 5

### Implementierung eines einfachen Embedding Layers ###

# es wird ein Zufallsstartwert (random seed) aus Gründen der Reproduzierbarkeit verwendet 
# Grund: Gewichte werden im Embedding Layer mit kleinen Zufallswerten initialisiert

torch.manual_seed(123)
embedding = torch.nn.Embedding(num_idx, out_dim)

# Print die Gewichte des Embedding Layers
print(embedding.weight)

# Print + Konvertierung aller Trainigsbeispiele
# der Ausdruck wird per Index unser Eingaben ausgedruckt
print(embedding(idx)) 




### Implementierung eins nn.Linear Layers ###
# Es soll gezeigt werden, dann der obige Embedding Layers in PyTorch genaus dasselbe leistet
# wie der nn.Linear-Layer in einer One-Hot-kodierten Darstellung. 
'''
Bei der One-Hot-Kodierung werden kategorische Daten in mehrdimensionale Binärvektoren konvertiert.
Hinweis: Wenn die Dimensionaliät zunimmt, wird das Training langsamer und komplexer. One-Hot benötigt auch mehr
Speicherplatz. 
'''
onehot = torch.nn.functional.one_hot(idx)
print(onehot)

# Inizialisierung des Linear Layers - mit der Matrixmultiplication XW^T
torch.manual_seed(123)
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
print(linear.weight)

# zum Vergleich werden die selben Gewichte vom Embedding Layer dem Linear Layer zugewiesen
linear.weight = torch.nn.Parameter(embedding.weight.T)

# Anwendung des Linear Layers auf die One-hot-codierte Darstellung der Eingaben
# und Vergleich
print("\n\n")
print(linear(onehot.float()))
print(embedding(idx))

