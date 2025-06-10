import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import tiktoken
import tensorflow
import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from c5_gpt_download import download_and_load_gpt2
from c5_pretraining_unlabeled_data import GPTModel, load_weights_into_gpt, generate_text_simple, text_to_token_ids, token_ids_to_text
import time
import matplotlib.pyplot as plt


'''
Kapitel 6:
Hier geht es um das Finetuning. Man kann in zwei Methoden unterscheiden: 
Instruction Finetuning (Anweisungen) und Classification Finetuning (Klassifizierung)

Es wird sich das Classification Finetuning angeschaut: 
- Vorbereitung des Datensatzes
- Setup vom Model
- Finetuning und Nutzung des Models
'''

### Klassen ###
### 6.3. Erstellen eines Data Loaders ###
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize Text
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Sequenzen abschneiden, wenn sie länger als max_length sind
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # auffüllen der Seqeunzen bis zur längsten Sequenz
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)



### 6.7 Finetuning des Models mit überwachten Daten (supervised data) ###
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Inizialisierung von Listen, um die Verluste bei den Beispielen zu verfolgen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Model wird in den Training Modus versetzt

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Verlustgradienten aus der vorherigen Batch-Iteration zurücksetzen
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Verlustgradienten berechnen
            optimizer.step() # Aktualisieren Sie Modellgewichte mithilfe von Verlustgradienten
            examples_seen += input_batch.shape[0] # Nachverfolgung von Beispielen anstatt von Tokens
            global_step += 1

            # optionaler Überprüfungsschritt
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Berechnen Sie die Genauigkeit nach jeder Epoche
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


### Funktionen ###

### 6.2 Vorbereitung der Daten ###
# Download des Datensets
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Herunterladen von den Daten
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Entzippen von den Daten
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # inzufügen von einem .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")



# Zur Erstellung eines ausbalancierten Datensat
def create_balanced_dataset(df):
    
    # Zählt die Instanzen von "Spam"
    num_spam = df[df["Label"] == "spam"].shape[0] 
    
    # Stichproben von „Ham“-Instanzen, um die Anzahl der „Spam“-Instanzen abzugleichen
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Kombinierung von Ham-"Subset" mit "Spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df



# Aufteilung der Daten in Trainings-, Validierungs- und Testdatensätzen
# Aufteilung des Datensatzes in 70% Training, 10% Validierung, 20% Testing
def random_split(df, train_frac, validation_frac):
    # Mischen des gesamten DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Berechnung der Split-Indizies
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Aufteilung des DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

### 6.6 Berechnung des Klassifizierungsverlusts und der Klassifizierungsgenauigkeit ###
# Berechnung der Klassifizierungsgenauigkeit
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits vom letzten Output Token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


# Berechne Batch Verluste 
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits vom letzten Output Token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Berechnung des Verlusts über alle
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


### 6.7 Finetuning des Models mit überwachten Daten (supervised data) ###
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# Grafische Darstellung der Verlust-Funktion für die Trainings- und Validierung Datensets
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Trainings- und Validierungsverluste im Vergleich zu Epochen darstellen
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Erstellt eine zweite x-Achse für die Beispiele
    ax2 = ax1.twiny()  # Erstellt eine zweite x-Achse welche die selbe x-Achse benutzt
    ax2.plot(examples_seen, train_values, alpha=0)  # Unsichtbares Diagramm zum Ausrichten von Haken
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Passen Sie das Layout an, um Platz zu schaffen
    plt.savefig(f"{label}-plot.pdf")
    plt.show()



### 6.8 Verwendung des LLM als Spam-Klassifizierer ###


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Vorbereiten der Eingaben für das Modell
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Kürzen Sie Sequenzen, wenn sie zu lang sind
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Füllt die Sequenzen bis zur längsten Sequenz auf
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Hinzufügen einer Batch Dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits vom letzten Output Token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Geben Sie das klassifizierte Ergebnis zurück
    return "spam" if predicted_label == 1 else "not spam"





### Tests und Ausführungen ###

def main():

    ### 6.2. Vorbereitung der Daten ###
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "BHT_BA2025_LLM_Raschka/sms_spam_collection.zip"
    extracted_path = "BHT_BA2025_LLM_Raschka/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 

    # Daten ansehen
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(df)

    print(df["Label"].value_counts())

    # Ausführung der Funktion create_balanced_dataset
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    # Änderung der Labels von ham und spam zu 0 und 1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})    
    print(balanced_df)

    # Ausführung der Funktion random_split
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("BHT_BA2025_LLM_Raschka/train.csv", index=None)
    validation_df.to_csv("BHT_BA2025_LLM_Raschka/validation.csv", index=None)
    test_df.to_csv("BHT_BA2025_LLM_Raschka/test.csv", index=None)


    ### 6.3  Erstellung des Data Loaders ###
    # Tokenizer inizialisieren 
    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


    # Trainings Datensatz erstellen
    train_dataset = SpamDataset(
        csv_file="BHT_BA2025_LLM_Raschka/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    # Validierungs Datensatz erstellen
    val_dataset = SpamDataset(
    csv_file="BHT_BA2025_LLM_Raschka/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
    )

    # Test Datensatz erstellen
    test_dataset = SpamDataset(
        csv_file="BHT_BA2025_LLM_Raschka/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    print(train_dataset.max_length)
    print(val_dataset.max_length)
    print(test_dataset.max_length)

    # Data Loader für die verschiedenen Datensets
    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # Testen des Trainings Data Loaders
    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")


    ### 6.4 Inizialisierung eines Models mit pre-trained Gewichten
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vokalegröße
        "context_length": 1024,  # Kontextlänge
        "drop_rate": 0.0,        # Dropout Rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval();

    # Prüfen ob alles richtig geladen wurde
    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))

    # Prüfen, ob das Model bereits eine Klassifizierung vor dem Finetuning vornehmen kann
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))


    ### 6.5 Hinzufügen eines Klassifizierungs Head ###
    # Das Model wird auf nicht trainierbar gesetzt
    for param in model.parameters():
        param.requires_grad = False

    # Der model.out_head wird nun ersetzt > von 50257 auf 2
    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    # Letzter Transformer Block wird auf Trainierbar gesetzt
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # Final LayerNorm wird auf trainierbar gesetzt
    for param in model.final_norm.parameters():
        param.requires_grad = True


    # mit den neuen Einstellungen > Text Input zum Testen
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

    
    ### 6.6 Berechnung des Klassifizierungsverlusts und der Klassifizierungsgenauigkeit ###
    # Letztes Token
    print("Last output token:", outputs[:, -1, :])

    # Konvertierung des Outputs (Logits) mithilfe der Softmax-Funktion in Wahrscheinlichkeitswerte
    # Ermittelung der Indexposition des größten Wahrscheinlichkeitswerts mit Argmax-Funktion
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    # Ausführung der Funktion calc_accuracy_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device) 

    torch.manual_seed(123) 

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


    # Verlustberechnung bei den Trainingsdaten
    with torch.no_grad(): 
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    ### 6.7 Finetuning des Models mit überwachten Daten (supervised data) ###
    '''
    # Training vom Model 
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    '''
    '''
    # Grafische Darstellung der Verlust-Funktion für die Trainings- und Validierung Datensets
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)


    # grafische Darstellung für die Verbesserung der Genauigkeit 
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    '''
    # Berechnung der Trainings-, Validierungs- und Testsatzleistungen über den gesamten Datensatz 
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # Testen ob die Klassifizierung nun funktioniert
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    
    # Speicherung das Models, damit wir es später aufrufen können, ohne es erneut trainieren zu müssen
    torch.save(model.state_dict(), "BHT_BA2025_LLM_Raschka/review_classifier.pth")


    # Damit kann man das gespeicherte trainierte Model aufrufen
    model_state_dict = torch.load("BHT_BA2025_LLM_Raschka/review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)


if __name__ == "__main__":
    main()
