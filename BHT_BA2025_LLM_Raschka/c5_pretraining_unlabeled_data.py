import torch
import tiktoken
from c4_implementing_gpt_model import GPTModel, generate_text_simple
import os
import urllib.request
from c2_data_preparation_sampling import create_dataloader_v1
import time
from c5_gpt_download import download_and_load_gpt2
import numpy as np

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

# Main Funktion für pretraining LLMs
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    # Inizialisiert Listen zum Nachverfolgen von Loss und Tokens
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Started den Main Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set model in den Trainingsmodus
                
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Verlustgradienten aus der vorherigen Batch-Iteration zurücksetzen
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Verlustgradienten berechnen
            optimizer.step() # Aktualisieren Sie Modellgewichte mithilfe von Verlustgradienten
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optionaler Auswertungsschritt
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Drucke nach jeder Epoche einen Beispieltext
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

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
    
    
    ### Variablen ###
    # GPT Model mit 124 Millionen Parameter
    GPT_CONFIG_124M = {

        "vocab_size": 50257,    #  # Vokabelgröße - in dem Fall 50527 Wörter
        "context_length": 256, # maximale Anzahl von Input Tokens das Model kann händeln via dem Positional Embeddings
        "emb_dim": 768,         # Größe des Embeddings - jeder Token wird in einen 768 dimensionalen Vektor umgewandelt
        "n_heads": 12,          # Anzahl der Attention Heads im Multi-head Attention Mechanismus
        "n_layers": 12,         # Anzahl von Transformer Blöcken
        "drop_rate": 0.1,       # Dropout Rate um Overfitting entgegenzuwirken - in dem Fall 10%
        "qkv_bias": False       # Bestimmte ob ein Bias-Vektor in den linearen Schichten des Multi-head Attention Mechanismus für Query-Key-Value Berchnungen mit einbezogen werden soll. 
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Dropout während der Inferenz deaktivieren

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Textgenerierung auf ein untrainiertes Modell
    
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    ### 5.1.2 Berechnung des Textgenierungsverlustes: Cross-entropy und Perplexity

    # An einem Beispiel: 
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                        [40, 1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]


    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Wahrscheinlichkeit jedes Tokens im Vokabular
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

    # die argmax Funktion gibt die Position des höchsten Wahrscheinlichkeitswertes in diesem Vektor zurück,
    # der die vorhergesagte Token-ID für das gegebene Token darstellt
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


    # Die den Zielindizes entsprechenden Token-Wahrscheinlichkeiten lauten wie folgt:
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    

    # Berechnen Sie den Logarithmus aller Token-Wahrscheinlichkeiten
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    # Berechnen Sie die durchschnittliche Wahrscheinlichkeit für jedes Token
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    

    # Logits haben folgenden Shape (batch_size, num_tokens, vocab_size)
    print("Logits shape:", logits.shape)

    # Targets haben folgenden Shape (batch_size, num_tokens)
    print("Targets shape:", targets.shape)

    # für die cross_entropy Funktion in PyTorch müssen die Tensoren abgeflacht werden, 
    # in dem sie über die Batch Dimension kombiniert werden. 
    # dabei werden batch_size * num_tokens berechnet.
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
        
    
    # die PyTorch cross_entropy Funktion kümmert sich automatisch darum, die Softmax- und Log-Wahrscheinlichkeitsberechnung intern 
    # auf die Token-Indizes in den Logits anzuwenden, die maximiert werden sollen.
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

    # Die Perplexität ist einfach die Exponentialfunktion des Kreuzentropieverlusts
    # Perplexität ist ein Maß dafür, wie gut die vom Modell vorhergesagte Wahrscheinlichkeitsverteilung mit 
    # der tatsächlichen Verteilung der Wörter im Datensatz übereinstimmt.
    perplexity = torch.exp(loss)
    print(perplexity)

    ### 5.1.3 Berechnung der Verluste im Trainings- und Validierungssatz ###
    # Laden der Daten
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # First 100 Buchstaben vom Text
    print(text_data[:99])

    # Last 100 Buchstaben vom Text
    print(text_data[-99:])

    # Analyse Text
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)


    # Aufteilung der Daten in Trainings und Validierungs Datenset
    train_ratio = 0.90 # 90% der Daten werden fürs Training genutzt, 10% für die Validierung
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Plausibilitätsprüfung

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")

    # Überprüfung die Daten richtig geladen wurden. 
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    
    # Prüfung, ob die Token-Größen im erwarteten Bereich liegen
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)



    # Anwendung der Funktion calc_loss_loader
    model.to(device) # kein zugewiesenes Modell = model.to(device), nötig für nn.Module classes


    torch.manual_seed(123) # Zur Reproduzierbarkeit

    with torch.no_grad(): # Deaktivieren Sie die Gradientenverfolgung aus Effizienzgründen, da noch nicht trainiert wird
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    
    ## Training der LLM ##
    # Zur Berechnung der Zeitdauer
    start_time = time.time()

    # Random Startwerte
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


    
    ### 5.3 Decoding Strategy zur Kontrolle von Zufälligkeiten
    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


    ### 5.3.3 Ausführung der neuen Textgenierungsfunktion "generate" ###
    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    ### 5.4 Laden und Speichern von Model weights in PyTorch ###
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "model_and_optimizer.pth"
    )

    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()

    ### 5.5 Laden von pretrained Weights von OpenAI
    BASE_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "drop_rate": 0.0,       # Dropout rate
        "qkv_bias": True        # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }


    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    file_name = "gpt2-small-124M.pth"
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    # file_name = "gpt2-xl-1558M.pth"

    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)
        print(f"Downloaded to {file_name}")

    gpt = GPTModel(BASE_CONFIG)
    gpt.load_state_dict(torch.load(file_name, weights_only=True))
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)


    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=BASE_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


    # Download der Model Weights von GPT-2
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    # Prüfung der Daten
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    # Kopieren Sie die Basiskonfiguration und aktualisieren Sie sie mit spezifischen Modelleinstellungen
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    load_weights_into_gpt(gpt, params)
    gpt.to(device)


    # Nutzung der Textgeneriung mit den neuen Einstellungen
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))




if __name__ == "__main__":
        main()