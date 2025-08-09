import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import json
import urllib
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
from c5_gpt_download import download_and_load_gpt2
from c5_pretraining_unlabeled_data_erweitert_wandb import GPTModel, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model
import time
from tqdm import tqdm
import re
import psutil
import urllib.request
import numpy as np
import random
import glob

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

# ----------------------------------------
# Dataset-Klasse & JSON-Lader - mit TinyStories
# ----------------------------------------
class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, text_key="story"):
        self.encoded_texts = []
        self.tokens = []
        for entry in data:
            if isinstance(entry, dict) and text_key in entry:
                toks = tokenizer.encode(entry[text_key])
                self.tokens.append(toks)
                if 0 < len(toks) <= max_length:
                    self.encoded_texts.append(toks)
        total_tokens = sum(len(t) for t in self.tokens)
        avg_tokens = total_tokens / len(self.tokens) if self.tokens else 0
        print(f"TinyStoriesDataset: {len(self.encoded_texts)} Einträge, "
              f"{total_tokens} Tokens gesamt, Ø {avg_tokens:.1f} Tokens/Eintrag.")

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

# ---------------------------
# DATEN-LADEFUNKTION
# ---------------------------
def load_tinystories_jsons(data_dir, max_files=None, max_entries=None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if max_files is not None:
        files = files[:max_files]
    data = []
    for fname in files:
        with open(fname, "r", encoding="utf-8") as f:
            try:
                d = json.load(f)
                if isinstance(d, list):
                    data.extend(d)
            except Exception as e:
                print(f"Warnung: Datei {fname} konnte nicht geladen werden: {e}")
    if max_entries is not None:
        data = data[:max_entries]
    print(f"load_tinystories_jsons: {len(data)} Einträge geladen aus {len(files)} Dateien.")
    return data

    
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


### 7.3 ###
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
            "seed": 5678,
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
    ### Zusatz: Variablen für TinyStories    
    max_files   = 4 # 4
    max_entries = 40000 # 40000 
    data_dir    = "test_test"
    data = load_tinystories_jsons(data_dir, max_files, max_entries)

    # Data Loader für Trainings-, Validierungs- und Testset
    num_workers = 0
    batch_size = 8 ### Zusatz: angepasst - original 8
    T = 1024 # Neue Variable für Sequence Length (wie bei Karpathy) -- geändert nun auf 512 

    # Aufteilung des Datensatzs in Trainings-, Validierungs- und Testdatensatz
    train_portion = int(len(data) * 0.85)  # 85% fürs Trainieren
    test_portion = int(len(data) * 0.1)    # 10% fürs Testen
    val_portion = len(data) - train_portion - test_portion  # 5% für die Validierung

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))


    ### 7.3 Organisieren von Daten in Trainingsbatches ###
    # Inizialisierung des Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    

    ### 7.4 Erstellen von Data Loader für einen Anweisungsdatensatz ###
    # Devicezuordnung
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length= T
    )

    ### Zusatz: Seed für Reproduzierbarkeit
    SEED = 123  # RUNS 123, 5678, 45887
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


    ### Zusatz: TinyStorie 
    # Dataset erstellen mit TinyStories
    if 'story' in data[0]:  # Check ob TinyStories Format
        train_dataset = TinyStoriesDataset(train_data, tokenizer)
        val_dataset = TinyStoriesDataset(val_data, tokenizer)
        test_dataset = TinyStoriesDataset(test_data, tokenizer)    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    ## Zusatz: Zum Auslesen wieviel Token wirklich nach Collat noch vorhanden sind. 
    total_train_tokens = 0
    for batch in train_loader:
        inputs, _ = batch
        total_train_tokens += inputs.numel()

    total_val_tokens = 0
    for batch in val_loader:
        inputs, _ = batch
        total_val_tokens += inputs.numel()

    print(f"RASCHKA Train tokens (nach Collate): {total_train_tokens:,}")
    print(f"RASCHKA Val tokens (nach Collate):   {total_val_tokens:,}")
    print(f"GESAMT: {total_train_tokens + total_val_tokens:,}")

    #import sys; sys.exit(0)


    ### 7.5 Laden des pretrained LLM mit 355 millionen parameter
    ### Zusatz BASE CONFIG custom angepasst, für kürzere Trainingsrunden
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,  # war vorher T wie Karpathy
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,     # wie Karpathy
        "n_layers": 12,      # wie Karpathy
        "n_heads": 12        # wie Karpathy
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"

    # BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    ### Zusatz BASE CONFIG custom angepasst, für kürzere Trainingsrunden
    
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    
    
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params) # Gewichte werden geladen
    model.to(device)
    model.eval()

    ### Zusatz: Optimizer und Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    ## Zusatz: zentralisierte Parameteranpassung
    num_epochs = 1
    lernrate = 5e-5
    eval_freq = 200
    eval_iter = 5
    warmup_steps = 800

    ### Zusatz: train_model_simple mit train_model
    start_time = time.time()
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        device=device,
        n_epochs=num_epochs, 
        eval_freq=eval_freq, # von 400 geändert, weil weniger Parameter können weniger Steps bedeuten
        eval_iter=eval_iter,
        start_context="Once upon a time", 
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        initial_lr=lernrate, # 3e-4
        min_lr= lernrate * 0.1,
        wandb_log=True,
        wandb_project="Analyse_auf_GPU",
        wandb_name=f"raschka_gpu_123",
        seed=SEED
    )
    
    print(f"Total steps completed: {len(train_loader) * num_epochs}")
    print(f"Total training batches: {len(train_loader)}")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Model speichern
    file_name = f"BHT_BA2025_LLM_Raschka/{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    







if __name__ == "__main__":
    main()