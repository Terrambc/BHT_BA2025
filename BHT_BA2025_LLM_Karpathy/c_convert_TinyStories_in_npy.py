import os
import json
import numpy as np
import tiktoken

def load_tinystories_jsons(data_dir, max_files=None, max_entries=None):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    if max_files is not None:
        files = files[:max_files]
    data = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
            except Exception as e:
                print(f"Fehler beim Laden von {fname}: {e}")
    if max_entries is not None:
        data = data[:max_entries]
    return data

def convert_to_token_array(data, tokenizer, key="story", max_length=1024):
    tokens = []
    for entry in data:
        if isinstance(entry, dict) and key in entry:
            text = entry[key]
            ids = tokenizer.encode(text)
            if 0 < len(ids) <= max_length:
                tokens.extend(ids + [tokenizer.eot_token])  # EOT für Trenner
    return np.array(tokens, dtype=np.int32)

def save_token_shard(tokens, split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split_name}_tokens.npy")
    np.save(path, tokens)
    print(f"Gespeichert: {path}, {len(tokens)} Tokens")

def main():
    data_dir = "test_test"
    output_dir = "BHT_BA2025_LLM_Karpathy/tinystory_npy"
    tokenizer = tiktoken.get_encoding("gpt2")

    data = load_tinystories_jsons(data_dir, max_files=1, max_entries=10000)
    assert len(data) > 0, "Keine Daten gefunden."

    # Aufteilen in Train/Val/Test
    ratio_train, ratio_val = 0.9, 0.1
    n = len(data)
    n_train = int(n * ratio_train)

    train_data = data[:n_train]
    val_data = data[n_train:]

    # Konvertierung und Speicherung
    train_tokens = convert_to_token_array(train_data, tokenizer)
    val_tokens = convert_to_token_array(val_data, tokenizer)

    save_token_shard(train_tokens, "train", output_dir)
    save_token_shard(val_tokens, "val", output_dir)

if __name__ == "__main__":
    main()