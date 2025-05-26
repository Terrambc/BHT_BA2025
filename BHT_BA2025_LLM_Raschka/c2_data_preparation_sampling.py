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

# Eine Tokenizer Klasse
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab # Speichert das Vokabular als Klassenattribut für den Zugriff in den Codier- und Decodiermethoden.
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    # Text wird in Tokens umgewandelt
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    # Tokens werden in Text umgewandelt    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Ersetzen Sie Leerzeichen vor den angegebenen Satzzeichen
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # Speichert das Vokabular als Klassenattribut für den Zugriff in den Codier- und Decodiermethoden.
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    # Text wird in Tokens umgewandelt
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Anpassung mit dem speziellen Token für unbekannte Wörter
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed] # Unbekannte Wörter/Zeichen werden mit <|unk|> ersetzt

        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    # Tokens werden in Text umgewandelt    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        # Ersetzen Sie Leerzeichen vor den angegebenen Satzzeichen
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text

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

    # Arbeitsdatei - Beispieltext herunterladen 
    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)


    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    ### 2.2 Tokenizing Text ###
    # Erster Text mit einem Beispieltest
    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    result = [item for item in result if item.strip()] # Leerzeichen und leere Strings entfernen mit List Comprehension
    
    print(result)

    # Tokenization vom raw_text (the-verdict)
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    # Berechnung der Anzahl von Token
    print(len(preprocessed))


    ### 2.3 Token in Token IDs konvertieren ###
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)

    # erstellt ein Dictionary mit den Token
    vocab = {token:integer for integer, token in enumerate(all_words)}

    # Abfrage der ersten 50 Einträge des Vokabulars
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

    # Aufruf der Klasse SimpleTokenizerV1
    # Encoding
    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    # Decoding
    print(tokenizer.decode(ids))

    # weitere Variante
    print(tokenizer.decode(tokenizer.encode(text)))

    ### 2.4 Speziellen Kontext Token hinzufügen ###

    # Was passiert, wenn an einen Text abfragt, den es in der Datei nicht gibt? 
    # Ein Fehler wird geworfen - die Vokabel "hello" gibt es nicht
    '''
    tokenizer = SimpleTokenizerV1(vocab)
    text = "Hello, do you like tea. Is this-- a test?"
    tokenizer.encode(text)
    '''
    
    # Spezielles Token für solche Fälle
    all_tokens = sorted(list(set(preprocessed))) # Liste mit allen eindeutigten Wörtern, alphabetisch sortiert
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}

    print(len(vocab.items()))

    # Zeigt die letzten fünf Vokabeln
    # <|endoftext|> und <|unk|> wurden mit Index 1130 und 1131 hinzugefügt
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)


    # Test nach der Klasse SimpleTokenizerV2
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

    ### 2.5 BytePair encoding ###
    tokenizer = tiktoken.get_encoding("gpt2") # Inizialisierung des GPT-2 Tokenizers
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
    )

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

    ### 2.6 Datenstichprobe mit Sliding Window ###
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Raw-Text wird in Tokens umgewandelt    
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))    

    # die ersten 50 Tokens im Datensatz
    enc_sample = enc_text[50:]

    # hier wird festgelegt viele Tokens in den Input einbezogen werden
    context_size = 4

    x = enc_sample[:context_size]
    # Das nächste Wort soll vorhergesagt werden, deswegen sind die Targets = Input + 1 (um eine Stelle nach rechts verschoben)
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:      {y}")

    # Erzeugung der input-target-pairs
    # Beispiel wie die Vorhersage als Tokenausgabe aussieht
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(context, "---->", desired)

    # Das ganze nochmal als Textausgabe (decode)
    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


    # Test der Funktion create_dataloader_v1

    # Anmerkung: stride=Zahl wird dazu genutzt, um zu bestimmen um wieviele Stellen das sliding window verschoben wird. 
    # batch_size > Anzahl der Tensoren der jeweiligen Kategorie (input, target), max_length > Anzahl von Token in einem Tensor, stride > um wieviel Stellen die Position verschoben wird. 
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    
    # Konvertiert den dataloader in einen Python-Iterator, um den nächsten Eintrag über die in Python integrierte Funktion next() abzurufen.
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    # Aufteilung in Inputs und Targets
    # in der Ausgabe sieht man, dass die Targets immer um eins nach rechts verschoben wurden
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


    ### 2.7 Erstellte Token Embeddings ###
    # Beispiel mit Input IDs 2,3,5,1 nach der Tokenisierung
    input_ids = torch.tensor([2, 3, 5, 1])

    # Beispiel Vokabular (klein) - 6 Wörter
    vocab_size = 6

    # Erzeugt die Embedding-Dimension - in dem Fall Größe 3
    output_dim = 3

    torch.manual_seed(123)
    # Erzeugt eine 6x3 Gewichtsmatrix (weight matrix)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    print(embedding_layer.weight)

    # Konvertiert den Token mit der ID 3 in einen 3-dimensionalen Vektor
    print(embedding_layer(torch.tensor([3])))

    # Einbettung aller vier Input_Ids-Werte
    print(embedding_layer(input_ids))


    ### 2.8. Encoding Wörterposition ###

    # Vokabelgröße
    vocab_size = 50257
    # 256-dimensionaler Vektor - Vorgabe
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    # Batchgröße ist 8 mit jeweils 4 tokens => Ergebnis wird sein ein 8 x 4 x 256 Tensor
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    # Token: zeigt wie das Embedding aussieht
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    print(token_embeddings)

    # Position: Zeigt wie die Embedding layer weights aussehen
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    print(pos_embedding_layer.weight)

    # Position: Zeigt wie die Embeeding Layers shapes und insgesamt aussehen
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)
    print(pos_embeddings)

    # Input: Um die in einem LLM verwendete Embeddings zu erstellen, werden Token und Position-Embedding kombiniert
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    print(input_embeddings)



if __name__ == "__main__":
    main()
