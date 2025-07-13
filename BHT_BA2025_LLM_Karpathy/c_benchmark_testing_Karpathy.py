import os
import time
import math
import torch
import wandb
import psutil
import tiktoken
from train_gpt2_nurCPU import GPT, GPTConfig, evaluate_mlm, DataLoaderLite


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
    device = torch.device("cpu")
    print("Using device:", device)
    torch.manual_seed(1337)

    wandb.init(project="BHT_Benchmark_2025", name="Karpathy_CPU_GPT_small")

    wandb.config.update({
    "total_batch_size": total_batch_size,
    "B": B,
    "T": T,
    "grad_accum_steps": grad_accum_steps,
    "max_lr": max_lr,
    "min_lr": min_lr,
    "warmup_steps": warmup_steps,
    "max_steps": max_steps,
    "weight_decay": 0.1,
    "vocab_size": config.vocab_size,
    "block_size": config.block_size,
    "n_layer": config.n_layer,
    "n_head": config.n_head,
    "n_embd": config.n_embd,
    })

    config = GPTConfig(
        vocab_size=50257,
        block_size=1024, 
        n_layer=12,
        n_head=12,
        n_embd=768,
    )

    total_batch_size = 8192  # entspricht batch_size=8 bei T=1024
    B, T = 8, 1024
    grad_accum_steps = total_batch_size // (B * T)
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 20
    # max_steps = 788 # Schritte für 1 Epoche (steps_per_epoch): 788 - errechnet in c_convert_TinyStories_npy_gleicherSatz
    max_steps = 849 # zweiter Versuch. Laut WandB.ai hat ein Epoch von Raschka nur 424 Schritte
    
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader   = DataLoaderLite(B=B, T=T, split="val")

    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Trainingsschleife
    for step in range(max_steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = B * T * grad_accum_steps / dt
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1e6

        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss = sum(model(x.to(device), y.to(device))[1].item()
                           for _ in range(2)
                           for x, y in [val_loader.next_batch()]) / 2

        val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
        train_ppl = math.exp(loss_accum.item()) if loss_accum.item() < 20 else float('inf')
        mlm_loss, val_acc = evaluate_mlm(model, val_loader, device, num_batches=1)
        mlm_loss, train_acc = evaluate_mlm(model, train_loader, device, num_batches=1)

        wandb.log({
            "step": step,
            "train_loss": loss_accum.item(),
            "val_loss": val_loss,
            "train_perplexity": train_ppl,
            "val_perplexity": val_ppl,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "batch_size": total_batch_size,
            "throughput_tokens_per_sec": tokens_per_sec,
            "memory_usage_MB": memory_usage,
        })

        print(f"Step {step:4d} | Train Loss: {loss_accum.item():.4f} | Val Loss: {val_loss:.4f} | Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | {tokens_per_sec:.1f} tok/s | Mem: {memory_usage:.1f} MB")

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

    print("Gesamt trainierte Tokens Karpathy:", max_steps * B * T)
    wandb.finish()

# ------------------------------------------------------
# Starte die Trainingsroutine
# ------------------------------------------------------
if __name__ == "__main__":
    main()
