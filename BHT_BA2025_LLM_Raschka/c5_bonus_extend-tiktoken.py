import tiktoken
import torch
from c5_pretraining_unlabeled_data import download_and_load_gpt2
from c4_implementing_gpt_model import GPTModel





def main():

    # Encoding einen neuen Text mit dem GPT2 Tokenizer
    base_tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "Hello, MyNewToken_1 is a new token. <|endoftext|>"

    token_ids = base_tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})
    print(token_ids)

    # Decode die Token IDs wieder
    for token_id in token_ids:
        print(f"{token_id} -> {base_tokenizer.decode([token_id])}")


    # Definition von einem Benutzerdefinierten Token und Token-IDs
    custom_tokens = ["MyNewToken_1", "MyNewToken_2"]
    custom_token_ids = {
        token: base_tokenizer.n_vocab + i for i, token in enumerate(custom_tokens)
    }

    

    # Erzeuge ein neues Encoding-Objekt mit erweiterten Token
    extended_tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
    )

    special_tokens_set = set(custom_tokens) | {"<|endoftext|>"}

    token_ids = extended_tokenizer.encode(
        "Sample text with MyNewToken_1 and MyNewToken_2. <|endoftext|>",
        allowed_special=special_tokens_set
    )
    print(token_ids)

    for token_id in token_ids:
        print(f"{token_id} -> {extended_tokenizer.decode([token_id])}")


    ### 2. Update von einem Pretrained LLM ###
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        
        "n_heads": 12,         
        "n_layers": 12,       
        "drop_rate": 0.1,      
        "qkv_bias": False      
    }

    # Definieren Sie Modellkonfigurationen in einem Wörterbuch für Kompaktheit
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Kopiert die Basiskonfiguration und aktualisieren sie mit spezifischen Modelleinstellungen
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval();     

    # Beispieltext - der mit dem ursprünglichen und dem neuen Tokenizer tokenisiert wird
    sample_text = "Sample text with MyNewToken_1 and MyNewToken_2. <|endoftext|>"

    original_token_ids = base_tokenizer.encode(
        sample_text, allowed_special={"<|endoftext|>"}
    )

    new_token_ids = extended_tokenizer.encode(
        "Sample text with MyNewToken_1 and MyNewToken_2. <|endoftext|>",
        allowed_special=special_tokens_set
    )

    # mit den alten Token-IDs    
    with torch.no_grad():
        out = gpt(torch.tensor([original_token_ids]))

    print(out)

    '''
    # mit den neuen Token-IDs
    with torch.no_grad():
        gpt(torch.tensor([new_token_ids]))

    print(out)
    '''

    print(gpt.tok_emb)

    # Erweiterung des Embedding Layesrs mit zwei neuen Einträgen
    num_tokens, emb_size = gpt.tok_emb.weight.shape
    new_num_tokens = num_tokens + 2

    # Erzeuge ein neuen Embedding Layer
    new_embedding = torch.nn.Embedding(new_num_tokens, emb_size)

    # Kopiere die Gewichte (weights) vom alten Embedding Layer
    new_embedding.weight.data[:num_tokens] = gpt.tok_emb.weight.data

    # Ersetze den alten Embedding Layer mit dem neuen im Model 
    gpt.tok_emb = new_embedding

    print(gpt.tok_emb)

    ### 2.4 Update des Output Layers ###
    print(gpt.out_head)

    # Erweiterung des Output Layers 
    original_out_features, original_in_features = gpt.out_head.weight.shape

    # Definierung einer neuen Anzahl von Output Features
    new_out_features = original_out_features + 2

    # Erstellung einer neuen Linearen Ebene mit der erweiterten Ausgabegröße (output size)
    new_linear = torch.nn.Linear(original_in_features, new_out_features)

    # Kopiere die Gewichte (weights) vom alten Linear Layer
    with torch.no_grad():
        new_linear.weight[:original_out_features] = gpt.out_head.weight
        if gpt.out_head.bias is not None:
            new_linear.bias[:original_out_features] = gpt.out_head.bias

    # Ersetze den alten Linear Layer mit dem neuen
    gpt.out_head = new_linear

    print(gpt.out_head)

    ### wie sieht das Ergebnis jetzt aus? ###
    with torch.no_grad():
        output = gpt(torch.tensor([original_token_ids]))
    print(output)
 

    with torch.no_grad():
        output = gpt(torch.tensor([new_token_ids]))
    print(output)






if __name__ == "__main__":
    main()