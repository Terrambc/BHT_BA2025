import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import json
import urllib
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
from c5_gpt_download import download_and_load_gpt2
from c5_pretraining_unlabeled_data import GPTModel, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text, calc_loss_loader, train_model_simple
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


### 7.3 ###
# Auffüllung der Sequenzen auf die maximale Länge der längsten Sequenz für die Inputwerte
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Sucht die längste Sequenz im Stapel
    # und erhöhe die maximale Länge um +1, 
    # wodurch ein zusätzliches Füllzeichen unten hinzugefügt wird 
    batch_max_length = max(len(item)+1 for item in batch)

    # Eingaben auffüllen und vorbereiten
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # hinzufügen eines <|endoftext|> token
        new_item += [pad_token_id]
        # Sequenzen auf batch_max_length auffüllen
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Über padded[:-1] wird das zusätzliche Padded-Token entfernt
        # das über die +1-Einstellung in batch_max_length hinzugefügt wurde
        # (das zusätzliche Fülltoken wird in späteren Codes relevant sein)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Konvertieren Sie die Liste der Eingaben in einen Tensor und übertragen Sie sie auf das Zielgerät
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor



# Verbesserte Version mit Targets und Inputs
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Finden der längsten Sequenz im Batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Eingaben und Targets auffüllen und vorbereiten
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
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Konvertiert die Liste der Inputs und Targets in einen Tensor und überträgt sie auf das Zielgerät
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


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

    # Laden vom Anweisungsdatensatz für dieses Kapitel
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    print("Example entry:\n", data[50])

    # Zeigen einer formatierten Antwort mit Eingabefeld - für Input mit Index 50
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"

    print(model_input + desired_response)

    # Das selbe für den Input mit Index 999
    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"

    print(model_input + desired_response)


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
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    # Testen der Funktion custom_collate_draft_1
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    print(custom_collate_draft_1(batch))

    # Ausführung der Funktion custom_collate_draft_2
    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)


    # Ausführung der Funktion custom_collate_fn
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    # Testen was diese neue ignore_index bewirkt

    # Beispiel 1 Ergebnis: tensor(1.1269)
    logits_1 = torch.tensor(
        [[-1.0, 1.0],  # 1st training example
        [-0.5, 1.5]]  # 2nd training example
    )
    targets_1 = torch.tensor([0, 1])
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)

    # Beispiel 2 wird im Ergebnis von 1 beeinflusst: Ergebnis: tensor(0.7936)
    logits_2 = torch.tensor(
        [[-1.0, 1.0],
        [-0.5, 1.5],
        [-0.5, 1.5]]  # New 3rd training example
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)

    # Beispiel 3 mit dem Einsatz vom index_ignore Wert, Ergebnis: tensor(1.1269)
    targets_3 = torch.tensor([0, 1, -100])

    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)


    ### 7.4 Erstellen von Data Loader für einen Anweisungsdatensatz ###
    # Devicezuordnung
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )


    # Data Loader für Trainings-, Validierungs- und Testset
    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    # Prüfen, ob die Inputs das <|endoftext|> padding Token mit der Token-ID 50256 haben
    print(inputs[0])

    # Prüfen ob die Targets das index_ignore Token haben
    print(targets[0])

    ### 7.5 Laden des pretrained LLM mit 355 millionen parameter
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vokabelgröße
        "context_length": 1024,  # Kontextlänge
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    # Prüfen wie es ohne Finetuning bei einer Validierungsaufgabe abschneidet
    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)


    # Kombinierung von Input und Output Text
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # Um die Antwort zu isolieren, wird die Länge der Anweisung vom Anfang des generierten Textes subtrahiert
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(response_text)


    ### 7.6 Finetuning das LLM mit Anweisungsdaten
    torch.manual_seed(123)

    # Berechnet den Verlust beim Trainings- und Validierungssatz
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    '''
    # Finetuning das Modell
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    

    ### 7.7 Extrahieren und Speichern von Antworten ###

    # Prüfen, wie die Antwort bei dem finetuned Model nun aussieht. 
    torch.manual_seed(123)
    for entry in test_data[:3]:

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
    )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")




    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text


    with open("BHT_BA2025_LLM_Raschka/instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

    
    # Test ob alles richtig eingetragen wurde
    print(test_data[0])

    # Speichern des Models für spätere Verwendung
    file_name = f"BHT_BA2025_LLM_Raschka/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    '''

    # Load model via
    # model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

    # Prüft ob Ollama gestartet wurde
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))


    #  prüfen mit Ollama
    file_path = "BHT_BA2025_LLM_Raschka/instruction-data-with-response.json"

    with open(file_path, "r") as file:
        test_data = json.load(file)

    



    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")


    # Erzeugt eine Bewertung zum Model = Score
    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")








if __name__ == "__main__":
    main()