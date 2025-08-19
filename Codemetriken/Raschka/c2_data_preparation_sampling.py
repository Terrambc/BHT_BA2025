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
# Erstellen Sie einen Datensatz und einen Dataloader, die Blöcke aus dem Eingabetextdatensatz (input text dataset) extrahieren
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # TTokenisiert den kompletten Text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Verwendet ein Sliding Window, um den Text in überlappende Sequenzen mit maximaler Länge aufzuteilen.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # return die Gesamtlänge der Zeilen im Datensatz
    def __len__(self):
        return len(self.input_ids)

    # Return eine einzelne Zeile im Datensatz
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


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
