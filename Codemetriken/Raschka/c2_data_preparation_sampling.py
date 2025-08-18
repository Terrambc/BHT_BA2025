import torch
import tiktoken
import os
import urllib.request
import re
import importlib
from torch.utils.data import Dataset, DataLoader

'''
Kapitel 2 Themen: 
- Word Embeddings verstehen
- Tokenizing Text
- Tokens in Token IDs konvertieren
- Speziellen Kontext Token hinzufügen
- BytePair Encoding
- Datenstichprobe mit Sliding Window
- Erstellung von Token Embeddings
- Encoding Wörterpositionen (Word Positions)
'''


### Klassen ###



### Funktionen ###

# Erzeugt einen Dataloader
def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize den Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Erzeugt einen Datensatz
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Erzeugt einen Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, # drop_last=True verwirft den letzten Batch, wenn er kleiner als die angegebene Batchgröße ist, um Verlustspitzen während des Trainings zu vermeiden.
        num_workers=num_workers # Die Anzahl der CPU-Prozesse, die für die Vorverarbeitung verwendet werden sollen
    )

    return dataloader




### Teste + Ausführung in Main() ###

def main():
    pass


if __name__ == "__main__":
    main()
