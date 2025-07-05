import os
import json
import glob
import time
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import tiktoken
import wandb
from c5_gpt_download import download_and_load_gpt2
from c5_pretraining_unlabeled_data import (
    GPTModel, load_weights_into_gpt, calc_loss_batch, calc_loss_loader,
    text_to_token_ids, token_ids_to_text, generate_and_print_sample
)
from c7_finetuning_follow_instruction_model import custom_collate_fn

# -------------------------------
# DATASET-KLASSE FÜR TINYSTORIES
# -------------------------------
class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, text_key="story"):
        self.encoded_texts = []
        self.tokens = []
        for entry in data:
            if isinstance(entry, dict) and text_key in entry:
                text = entry[text_key]
                tokens = tokenizer.encode(text)
                self.tokens.append(tokens)
                if 0 < len(tokens) <= max_length:
                    self.encoded_texts.append(tokens)
        total_tokens = sum(len(toks) for toks in self.tokens)
        avg_tokens = total_tokens / len(self.tokens) if self.tokens else 0
        print(f"TinyStoriesDataset: {len(self.encoded_texts)} gültige Einträge geladen.")
        print(f"TinyStories Gesamtanzahl Tokens: {total_tokens}")
        print(f"Durchschnittliche Tokenanzahl pro Eintrag: {avg_tokens:.2f}")
        

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.encoded_texts)

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

# ---------------------------
# ACCURACY-BERECHNUNG
# ---------------------------
def calculate_accuracy(logits, targets, ignore_index=-100):
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (preds == targets) & mask
    total = mask.sum()
    correct_count = correct.sum()
    if total.item() == 0:
        return 0.0
    return (correct_count.float() / total.float()).item()

# -----------------------------------------
# MINIMALISTISCHE BASIS-TRAININGSFUNKTION
# -----------------------------------------
def train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer
):
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
        # Optional: nach jeder Epoche ein Sample erzeugen
        generate_and_print_sample(model, tokenizer, device, start_context)
    # Kein Logging, keine Rückgabe – geeignet für Baselines!



# -----------------------------------------
# EXPERIMENTELLE TRAININGSFUNKTION MIT WANDB
# -----------------------------------------
def train_model_with_wandb(
    model, train_loader, val_loader, optimizer, device, total_epochs,
    eval_freq, eval_iter, start_context, tokenizer,
    project_name="TinyStories_Finetune", run_name="gpt2-medium-tinystories"
):
    wandb.init(project=project_name, name=run_name)
    best_val_loss = float('inf')
    best_model_path = "best_model.pt"
    global_step = 0

    for epoch in range(total_epochs):
        print(f"\nStarte Epoche {epoch+1}/{total_epochs}")
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            start_batch_time = time.time()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            # Logging & Metriken alle eval_freq Schritte
            if global_step % eval_freq == 0:
                # Losses und Perplexity
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                    val_loss   = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                    val_perplexity = torch.exp(torch.tensor(val_loss)).item() if val_loss < 20 else float('inf')
                    train_perplexity = torch.exp(torch.tensor(train_loss)).item() if train_loss < 20 else float('inf')
                    
                    # Accuracy auf Validation
                    acc_sum, acc_count = 0.0, 0
                    for i, (val_inputs, val_targets) in enumerate(val_loader):
                        if i >= eval_iter: break
                        val_logits = model(val_inputs.to(device))
                        acc = calculate_accuracy(val_logits, val_targets.to(device))
                        acc_sum += acc
                        acc_count += 1
                    val_acc = acc_sum / acc_count if acc_count > 0 else 0.0

                    # Accuracy auf Training
                    acc_sum2, acc_count2 = 0.0, 0
                    for i, (train_inputs, train_targets) in enumerate(train_loader):
                        if i >= eval_iter: break
                        train_logits = model(train_inputs.to(device))
                        acc = calculate_accuracy(train_logits, train_targets.to(device))
                        acc_sum2 += acc
                        acc_count2 += 1
                    train_acc = acc_sum2 / acc_count2 if acc_count2 > 0 else 0.0

                model.train()

                batch_size = input_batch.size(0)
                elapsed = time.time() - start_batch_time
                throughput = input_batch.numel() / elapsed if elapsed > 0 else 0
                if torch.cuda.is_available():
                    memory_usage = torch.cuda.memory_allocated(device) / 1e6
                else:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / 1e6  # RAM in MB

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                      f"Val pplx {val_perplexity:.3f}, Val_Acc {val_acc:.3f}, "
                      f"train pplx {train_perplexity:.3f}, Train_Acc {train_acc:.3f}")

                wandb.log({
                    "step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_perplexity": train_perplexity, 
                    "val_perplexity": val_perplexity,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "batch_size": batch_size,
                    "throughput_tokens_per_sec": throughput,
                    "memory_usage_MB": memory_usage,
                })


        # Nach jeder Epoche: Beispielgenerierung
        generate_and_print_sample(model, tokenizer, device, start_context)

    wandb.finish()

# -----------------------------------------
# HAUPTFUNKTION
# -----------------------------------------
def main():
    # ---- PARAMETER ANPASSEN ----
    data_dir = "test_test"  # Ordner mit deinen .json-Dateien
    batch_size = 8
    total_epochs = 1
    eval_freq = 10
    eval_iter = 5
    max_files = 4   # Für schnellen Test: z.B. 1
    max_entries = 40000 # Für schnellen Test: z.B. 100

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

    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    tokenizer = tiktoken.get_encoding("gpt2")

    # ---- DATEN LADEN UND AUFTEILEN ----
    data = load_tinystories_jsons(data_dir, max_files=max_files, max_entries=max_entries)

    assert len(data) > 0, "Keine Daten geladen!"

    train_ratio, val_ratio, test_ratio = 0.85, 0.10, 0.05
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_data = data[:n_train]
    val_data   = data[n_train:n_train + n_val]
    test_data  = data[n_train + n_val:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ---- DATASET UND DATALOADER ----
    train_dataset = TinyStoriesDataset(train_data, tokenizer)
    val_dataset   = TinyStoriesDataset(val_data, tokenizer)
    test_dataset  = TinyStoriesDataset(test_data, tokenizer)

    import sys; sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    collate = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    # ---- MODELL ----
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # ---- TRAINING mit wandb ----
    start_context = "Once upon a time,"
    train_model_with_wandb(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        total_epochs=total_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context=start_context,
        tokenizer=tokenizer,
        project_name="BHT_Benchmark_2025",
        run_name="Raschka_CPU_GPT_small"
    )

    # ---- Optional: MINIMALISTISCHES TRAINING OHNE LOGGING ----
    # train_model_simple(
    #     model, train_loader, val_loader, optimizer, device, 1,
    #     eval_freq, eval_iter, start_context, tokenizer
    # )

    # ---- Beispiel-Generierung ----
    prompt = "Once upon a time,"
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        from c5_pretraining_unlabeled_data import generate
        token_ids = generate(model, idx, max_new_tokens=50, context_size=BASE_CONFIG["context_length"])
        gen_text = token_ids_to_text(token_ids, tokenizer)
    print("\nPrompt:", prompt)
    print("Generated:", gen_text[len(prompt):].strip())

# -----------------------------------------
if __name__ == "__main__":
    main()
