import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import json
import urllib
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
from c5_gpt_download import download_and_load_gpt2
# from c5_pretraining_unlabeled_data import GPTModel, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model_simple
from c5_pretraining_unlabeld_data_erweitert import GPTModel, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model
import time
from tqdm import tqdm
import re
import psutil
import urllib.request


'''
Kapitel 7: Finetuning damit das Model Anweisungen befolgen kann

- Dataset Download und Vorbereitung
- Batching vom Datensatz
- Erstellen von Data Loader
- Laden von einem pretrained LLM
- Anweisung Finetuning die LLM
- Inspektion der Modellierungsverlust
- Antworten extrahieren
- qualitative Bewertung
- Bewertung der Antwort

'''

### Klassen ###
### 7.3. Organisieren von Daten in Trainingsbatches ###

# Alle Inputs datensatz werden pre-tokenizes
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)




### Funktionen ###

## 7.2 Vorbereiten eines Datensatzes für die überwachte (supervised) Feinabstimmung von Anweisungen ##
# Download es Anweisungsdatensatz für diese Kapitel
def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data



# Es wird hier das Alpaca-style Format genutzt: Instruction - Input - Response
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# Final Version: mit einem "ignore_index"-Wert, um alle Padding-Token-IDS(50256) (Gilt nur für die Targets)
# durch einen neuen Wert zu ersetzen. 
# Zweck: mit diesem neuen Wert, können wir die Padding-Werte in der Verlustfunktion ignorieren
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Suchen Sie die längste Sequenz im Batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Inputs und Targets auffüllen und vorbereiten
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # hinzufügen eines <|endoftext|> token
        new_item += [pad_token_id]
        # auffüllen der Sequenz zu max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Das letzte Token für Inputs abschneiden
        targets = torch.tensor(padded[1:])  # Shift +1 nach rechts für Targets

        # Ersetze alle Fülltoken außer dem ersten in Targets durch ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optional auf die maximale Sequenzlänge kürzen
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Konvertiert die Liste der Inputs und Targets in Tensoren und überträgt sie auf das Zielgerät
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# Prüft nach ob Ollama gestartet ist
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


# Aufruf des gespeicherten Modells
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# Um Ollama mit der REST API in Python zu starten
def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
   
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Für deterministische Antworten sind die folgenden Einstellungen erforderlich
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Konvertiert das Wörterbuch in eine JSON-formatierte Zeichenfolge und kodiert es in Bytes
    payload = json.dumps(data).encode("utf-8")

    # Erstellt ein Anforderungsobjekt, setzt die Methode auf POST und fügt die erforderlichen Header hinzu
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Sendet die Anfrage und erfasst die Antwort
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Liest und Decodiert die Antwort
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


# Erzeugt ein Score für das Model
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores



### Test und Ausführungen ###
def main():

    pass








if __name__ == "__main__":
    main()