"""
PRISMA-konformes Preprocessing-Skript:
Erstellt aus einer JSON-Datenbasis Train-/Val-/Test-NPY-Shards mit 85/10/5-Split,
begrenzt auf 40.000 Einträge, max_length=1024 je Eintrag,
und exakt definierter Gesamtanzahl Trainings-Tokens.
Unterstüzung durch KI Claude Opus4.
"""

import os
import json
import numpy as np
import tiktoken
import random
import math

# ---- Parameter ----
json_files = [
    "test_test/data00.json",
    "test_test/data01.json",
    "test_test/data02.json",
    "test_test/data03.json"
]
output_dir = "BHT_BA2025_LLM_Karpathy/tinystory_npy_vergleichbar"
os.makedirs(output_dir, exist_ok=True)

train_ratio = 0.85
val_ratio = 0.05
test_ratio = 0.1
shard_size = 100000        # Karpathy-Format
max_length = 512         # Max. Länge pro Eintrag #vorher 1024
train_token_limit = 6449511   # Exakte Anzahl Trainings-Tokens wie bei Raschka

tokenizer = tiktoken.get_encoding("gpt2")
text_key = "story"        # ggf. anpassen

B = 4        # Batchgröße (wie im Karpathy-Training) # vorher 8
T = 512     # Kontextlänge (wie im Karpathy-Training) # vorher 1024

# ---- 1. Einlesen und Mischen der Daten ----
data = []
for fpath in json_files:
    with open(fpath, "r", encoding="utf-8") as f:
        data.extend(json.load(f))
print(f"Gesamtzahl Einträge (vor Filter): {len(data)}")

# Optional: Filter auf gültige Einträge mit Key "story"
data = [entry for entry in data if isinstance(entry, dict) and text_key in entry]
print(f"Gesamtzahl Einträge (nach Filter): {len(data)}")

# ---- 2. Auf 40.000 Einträge begrenzen ----
random.seed(123)
random.shuffle(data)  # Reproduzierbare Zufallsaufteilung

if len(data) > 40000:
    data = data[:40000]
    print(f"Begrenzung: Es werden genau {len(data)} Einträge verwendet.")

# ---- 3. Aufteilung in Train/Val/Test ----
n_total = len(data)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ---- 4. Tokenisierung mit max_length-Filter und Tokenlimit für Training ----
def tokenize_entries_limit_tokens(entries, tokenizer, key, max_length, token_limit=None):
    tokens = []
    included = 0
    for entry in entries:
        entry_tokens = tokenizer.encode(entry[key])
        if 0 < len(entry_tokens) <= max_length:
            if token_limit is not None and len(tokens) + len(entry_tokens) > token_limit:
                # Letzter Eintrag ggf. abschneiden
                remaining = token_limit - len(tokens)
                tokens.extend(entry_tokens[:remaining])
                included += 1
                break
            tokens.extend(entry_tokens)
            included += 1
    return tokens, included

# Train: Tokenlimit setzen
train_tokens, train_included = tokenize_entries_limit_tokens(train_data, tokenizer, text_key, max_length, token_limit=train_token_limit)
# Val/Test: Kein Tokenlimit, aber max_length-Filter
val_tokens, val_included     = tokenize_entries_limit_tokens(val_data, tokenizer, text_key, max_length)
test_tokens, test_included   = tokenize_entries_limit_tokens(test_data, tokenizer, text_key, max_length)

print(f"\nTokenanzahl Train: {len(train_tokens)} (Einträge: {train_included})")
print(f"Tokenanzahl Val:   {len(val_tokens)} (Einträge: {val_included})")
print(f"Tokenanzahl Test:  {len(test_tokens)} (Einträge: {test_included})")

# ---- 4.5 Batch-konformes Padding am Shard-Ende ----
btc = B * T + 1
for name, tokens in zip(['train', 'val', 'test'], [train_tokens, val_tokens, test_tokens]):
    rem = len(tokens) % btc
    if rem:
        pad_len = btc - rem
        tokens.extend(tokens[-pad_len:])
        print(f"Padded {name}_tokens um {pad_len} → {len(tokens)} Tokens")



# ---- 5. Schritte pro Epoche (für Karpathy) berechnen ----
steps_per_epoch = math.ceil(len(train_tokens) / (B * T))
print(f"Steps pro Epoche: {steps_per_epoch}")

# ---- 6. Sharding und Export ----
def write_npy_shards(tokens, prefix, shard_size, outdir, min_size=2048):
    num_shards = (len(tokens) + shard_size - 1) // shard_size
    for i in range(num_shards):
        shard_tokens = tokens[i * shard_size : (i+1) * shard_size]

    # Padding für die letzte Shard
        if len(shard_tokens) < min_size and i == num_shards - 1:
            # Wiederhole die letzten Tokens
            pad_needed = min_size - len(shard_tokens)
            padding = shard_tokens[-pad_needed:] if len(shard_tokens) >= pad_needed else shard_tokens * (pad_needed // len(shard_tokens) + 1)
            shard_tokens = list(shard_tokens) + list(padding[:pad_needed])
            print(f"Padded last {prefix} shard from {len(shard_tokens)-pad_needed} to {len(shard_tokens)} tokens")

        
        npy_path = os.path.join(outdir, f"{prefix}_{i:02d}.npy")
        np.save(npy_path, np.array(shard_tokens, dtype=np.int32))

# Minimum size für einen Batch
min_size = B * T + 1  # = 2048

write_npy_shards(train_tokens, prefix="tinystory_train", shard_size=shard_size, outdir=output_dir, min_size=min_size)
write_npy_shards(val_tokens,   prefix="tinystory_val",   shard_size=shard_size, outdir=output_dir, min_size=min_size)
write_npy_shards(test_tokens,  prefix="tinystory_test",  shard_size=shard_size, outdir=output_dir, min_size=min_size)

print(f"\nFinale Tokenzahlen:")
print(f"Train: {len(train_tokens)}")
print(f"Val:   {len(val_tokens)}")
print(f"Test:  {len(test_tokens)}")
print("Alle NPY-Shards wurden erzeugt und können direkt von Karpathy genutzt werden.")
