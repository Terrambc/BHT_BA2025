import torch
import tiktoken
from c4_implementing_gpt_model import GPTModel, generate_text_simple
import os
import urllib.request
from c2_data_preparation_sampling import create_dataloader_v1
import time
from c5_gpt_download import download_and_load_gpt2
import numpy as np
import math
import wandb

'''
Im Kapitel 5 geht es um das vortrainieren eines LLM. 
Dabei wird die Trainingsschleife und der Code für die grundlegende
Modelbewertung implementiert. 

* Pretraining
* Trainingsschleife
* Modelbewertung (model evaluation)
* Laden von Pretrained Weights

Themen: 
- Textgenerierung
- Textbewertung (Evaluation)
- Training sets und Validation Sets Verluste
- LLM Trainings Funktion
- Textgenerierungsstrategie
- Gewichte speichern und laden
- Pretrained weights von Open AI laden

'''

### Klassen ###




### Funktionen ###
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Fügt Batch Dimension hinzu
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # entfernt Batch Dimension
    return tokenizer.decode(flat.tolist())

# Berechnt den Verlust (loss) für einen einzelnen Stack
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# Berechnet den Verlust (loss) für alle Stacks
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) #Iteriert über alle Batches, wenn spezifisches num_batches benannt wurde
    else:
        # Reduziert die Anzahl der Batches, um die der Gesamtanzahl der Batches im Dataloader anzupassen, falls sonst die Anzahl überschritten wird
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() # Summiert lden Verlust für jeden Batch
        else:
            break
    return total_loss / num_batches # Berechnet den Verlustdurchschnitt für alle Batches




### 5.2 Training eines LLM ###

# Main Funktion für pretraining LLMs - jetzt mit warmup, gradient Accumulation und cosine decay 
def train_model(model, train_loader, val_loader, optimizer, device, n_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, warmup_steps, initial_lr=3e-05, min_lr=1e-6):

    # Inizialisiert Listen zum Nachverfolgen von Loss und Tokens
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], [] 
    tokens_seen, global_step = 0, -1

    ### NEU: Inizialisierung für Warmup (Aufwärmphase)
    peak_lr = optimizer.param_groups[0]["lr"] # ruft die inizial Lernrate vom Optimizer ab
    total_training_steps = len(train_loader) * n_epochs # Berechnet die Komplettanzahl von Iterationen im Trainingsprozess
    lr_increment = (peak_lr - initial_lr) / warmup_steps # Berechnet die Lernratensteigerung während der Aufwärmphase

    # startet den Main  Training Loop
    for epoch in range(n_epochs):
        model.train() # Versetzt das Model in den Trainingsmodus        
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Verlustgradienten aus der vorherigen Batch-Iteration zurücksetzen
            global_step += 1

            if global_step < warmup_steps: # Passt die Lernrate an die aktuelle Phase an (Aufwärmphase oder Cosinus-Decay).
                lr = initial_lr + global_step * lr_increment
            else:
                # Cosine Decay nach der Aufwärmphase
                progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups: # Wendet die berechnete Lernrate auf den Optimierer an
                param_group["lr"] = lr
            track_lrs.append(lr) # speichert die aktuelle Lernrate

            # Berechnet und Backproagate den Verlust
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Wendet Gradienten-Clipping nach der Aufwärmphase an, um explodierende Gradienten zu vermeiden.
            if global_step > warmup_steps: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else: 
                # Der Code im Buch hat ursprünglich global_step > warmup_steps, was zum überspringen von clipping führt, nach der Aufwärmphase
                if global_step >= warmup_steps:  
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)                

            optimizer.step()
            tokens_seen += input_batch.numel()

            # Optionaler Auswertungsschritt
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # Drucke nach jeder Epoche einen Beispieltext
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, track_lrs

# Berechnung von Verlsuten über den Trainings- und Validierungssatz
# stellt dabei sicher, dass sich das Model im Evaluationsmodus befindet 
# Gradient und Dropout sind deaktiviert
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Dropout ausgeschaltet

    # schaltet das Gradient Tracking aus
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Nachverfolgung ob sich das Model während des Trainings verbessert
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # Dropout ausgeschaltet
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # schaltet das Gradient Tracking aus
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=100, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


### 5.3.1 + 5.3.2 Temperature scaling und Top-k sampling ###
### 5.3.3 Modifizierung der Textgeneriungsfunktion ###
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # #Schleife - holt Logits und fokusiert nur auf den letzten Schritt
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filtert logits mit top_k Auswertung
        if top_k is not None:
            # behalt nur die Top-K Werte
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Fügt Temperature scaling hinzu
        if temperature > 0.0:
            logits = logits / temperature

            # Wenden Sie Softmax an, um Wahrscheinlichkeiten zu erhalten
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Stichprobe aus der Verteilung
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Ansonsten wie zuvor: IDX des Vokabeleintrags mit dem höchsten Logit-Wert abrufen
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Beendet die Generierung vorzeitig, wenn ein Sequenzende-Token gefunden wird und eos_id angegeben ist
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

# OpenAI-Gewichte den entsprechenden Weight Tensoren meines GPTModel Instanz zuzuweisen
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


### Tests und Ausführungen ###
def main():
    
    
    pass




if __name__ == "__main__":
        main()