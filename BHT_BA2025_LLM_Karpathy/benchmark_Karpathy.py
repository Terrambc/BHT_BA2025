#!/usr/bin/env python3
"""
benchmark_karpathy.py

Benchmark-Skript für Karpathys GPT-2-Implementierung (train_gpt2_nurCPU.py)
mit originalem DataLoaderLite und Weights & Biases Logging.

Loggt folgende Metriken:
  - step
  - train_loss
  - val_loss
  - train_perplexity
  - val_perplexity
  - val_accuracy
  - batch_size
  - throughput_tokens_per_sec
  - memory_usage_MB


"""

import math
import time
import torch
import wandb
import psutil
import random
import numpy as np
import tiktoken

from train_gpt2_nurCPU import GPT, GPTConfig, DataLoaderLite, evaluate_mlm

# ------------------------------------------------------
# Metriken: Memory und Lernrate
# ------------------------------------------------------

def get_memory_usage():
    proc = psutil.Process()
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return proc.memory_info().rss / 1e6



def get_lr(step, *, max_learning_rate, min_learning_rate, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


# ------------------------------------------------------
# Funktion zur Textgenerierung aus einem Startprompt
# ------------------------------------------------------
def generate(model, idx, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids[0].tolist())

def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text)
    return torch.tensor([ids], dtype=torch.long)



# ------------------------------------------------------
# Hauptfunktion: Training mit wandb und DataLoaderLite
# ------------------------------------------------------
def main():
    # — Hyperparameter —
    B               = 4                 # Micro-Batch-Size # vorher 8
    T               = 512              # Kontextlänge # vorher 1024
    total_batch     = 2048              # effektive Batch-Größe # vorher 8192
    grad_accum      = total_batch // (B * T)
    max_steps       = 4250 # versuch ob es immer noch klappt oder doch 788 ----- scheinbar habe ich vorher für steps die Zeilen getrackt - 849 entspricht zwei Epochs bei Raschka
    eval_interval   = 10
    eval_iters      = 5
    max_learning_rate  = 5e-4
    min_learning_rate  = max_learning_rate * 0.1
    warmup_steps    = 20


    # — Seed Setup —
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    # Listen für intra-run-Variabilität
    val_losses_over_time = []
    val_accs_over_time   = []

    # — Weights & Biases Init —
    wandb.init(
        project="vergleich-raschka-karpathy",
        name="karpathy-gpt2-small",
        config={
            "batch_size":           B,
            "effective_batch_size": total_batch,
            "context_length":       T,
            "grad_accum_steps":     grad_accum,
            "max_steps":            max_steps,
            "eval_interval":        eval_interval,
            "eval_iters":           eval_iters,
            "max_learning_rate":    max_learning_rate,
            "weight_decay":         0.1,
            "warmup_steps":         warmup_steps,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    wandb.config.update({
        "device":      str(device),
        "cpu_cores":   psutil.cpu_count(logical=False),
        "total_ram_GB": psutil.virtual_memory().total / (1024**3),
        **({"gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda}
           if torch.cuda.is_available() else {})
    })

    # — Modell & Optimizer —
    config = GPTConfig(
        vocab_size=50257,
        block_size=T,
        n_layer=6, # vorher 12
        n_head=6,  # vorher 12
        n_embd=384, # vorher 768
    )
    model     = GPT(config).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_learning_rate
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    # — DataLoaderLite —
    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader   = DataLoaderLite(B=B, T=T, split="val")

    # — Trainingsschleife mit Logging —
    for step in range(max_steps):
        # Training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        t0 = time.time()
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum
            loss_accum += loss.detach().item()
            loss.backward()
        # Gradient clipping & LR scheduling
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step,
                    max_learning_rate=max_learning_rate,
                    min_learning_rate=min_learning_rate,
                    warmup_steps=warmup_steps,
                    max_steps=max_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        t1 = time.time()

        # Durchsatz & Speicher
        tokens_processed = B * T * grad_accum
        dt_train         = t1 - t0
        throughput       = tokens_processed / dt_train if dt_train > 0 else 0.0
        memory_usage     = get_memory_usage()

        # Evaluation & Wandb-Logging
        if step % eval_interval == 0 or step == max_steps - 1:
            # Validierung
            val_loader.reset()
            val_loss, val_acc = evaluate_mlm(model, val_loader, device, num_batches=eval_iters)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')

            val_losses_over_time.append(val_loss)    
            val_accs_over_time.append(val_acc)

            # train_accuracy über evaluate_mlm auf dem Train-Split
            train_loader.reset()
            train_loss, train_acc = evaluate_mlm(model, train_loader, device, num_batches=eval_iters)
            train_ppl  = math.exp(train_loss) if train_loss < 20 else float('inf')
            

            wandb.log({
                "step":                          step,
                "train_loss":                    train_loss,
                "val_loss":                      val_loss,
                "train_perplexity":              train_ppl,
                "val_perplexity":                val_ppl,
                "val_accuracy":                  val_acc,
                "train_accuracy":                train_acc,
                "batch_size":                    B,
                "throughput_tokens_per_sec":     throughput,
                "memory_usage_MB":               memory_usage,
            }, step=step)

            print(f"Step {step:4d} | "
                  f"train_loss {train_loss:.4f}, val_loss {val_loss:.4f} | "
                  f"train_ppl {train_ppl:.1f}, val_ppl {val_ppl:.1f} | "
                  f"train_acc {train_acc:.3f}, val_acc {val_acc:.3f} | tok/s {throughput:.1f} | mem {memory_usage:.0f}MB")


            # Beispielgenerierung nach 1000 Schritten
            if step == max_steps - 1:
                prompt = "Once upon a time,"
                idx = text_to_token_ids(prompt, tokenizer).to(device)
                with torch.no_grad():
                    token_ids = generate(model, idx, max_new_tokens=100, context_size=config.block_size)
                    gen_text = token_ids_to_text(token_ids, tokenizer)
                print("\nPrompt:", prompt)
                print("Generated:", gen_text[len(prompt):].strip())

    # Abschließende Beispielgenerierung
    prompt = "Once upon a time,"
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, idx, max_new_tokens=100, context_size=config.block_size)
        gen_text = token_ids_to_text(token_ids, tokenizer)
    print("\n[Final Sample after Training]")
    print("Prompt:", prompt)
    print("Generated:", gen_text[len(prompt):].strip())

    loss_mean = np.mean(val_losses_over_time)
    loss_std  = np.std(val_losses_over_time, ddof=1)
    acc_mean  = np.mean(val_accs_over_time)
    acc_std   = np.std(val_accs_over_time,    ddof=1)

    wandb.log({
        "val_loss_mean":     loss_mean,
        "val_loss_std":      loss_std,
        "val_accuracy_mean": acc_mean,
        "val_accuracy_std":  acc_std,
    })

    # — Finale Evaluation —
    train_loader.reset(); val_loader.reset()
    final_val_loss, final_val_acc = evaluate_mlm(model, val_loader, device, num_batches=eval_iters)
    final_val_ppl = math.exp(final_val_loss) if final_val_loss < 20 else float('inf')

    final_train_loss, final_train_acc = evaluate_mlm(model, train_loader, device, num_batches=eval_iters)
    final_train_ppl = math.exp(final_train_loss) if final_train_loss < 20 else float('inf')

    wandb.log({
        "final_val_loss":     final_val_loss,
        "final_val_perplexity": final_val_ppl,
        "final_val_accuracy": final_val_acc,

        "final_train_loss": final_train_loss,
        "final_train_perplexity": final_train_ppl,
        "final_train_accuracy": final_train_acc,
       
        "total_training_steps": step,
        "total_tokens_seen": step * total_batch,

    })


    wandb.finish()

if __name__ == "__main__":
    main()
