import torch
from thop import profile

from c4_implementing_gpt_model import GPTModel


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)




### Simple benchmark with fixed batch size ###

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    model = GPTModel(BASE_CONFIG).bfloat16()
    model.to(device)

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)

    # FLOPs (Floating Point Operations Per Second) messen die Rechenkomplexität neuronaler Netzwerkmodelle, 
    # indem sie die Anzahl der ausgeführten Gleitkommaoperationen zählen
    flops = 2*macs
    print(f"{size:18}: {flops:.1e} FLOPS")

    del model
    torch.cuda.empty_cache()


### Simple benchmark with automatic batch size finding ###
for size in model_configs:
    print(f"\nProcessing {size}")
    config = BASE_CONFIG.copy()
    config.update(model_configs[size])

    min_batch_size = 1
    max_batch_size = None
    max_possible_batch_size = 4096

    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            input_tensor = torch.randint(
                0, config["vocab_size"],
                (batch_size, config["context_length"]),
                device=device
            )

            model = GPTModel(config).bfloat16().to(device)

            # MACS = multiply-accumulate operations
            # MACS are typically counted as two FLOPS (one multiply and one accumulate)
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops = 2 * macs
            print(f"  Batch size {batch_size}: {flops:.1e} FLOPS")

            # If successful, try a larger batch size
            min_batch_size = batch_size + 1
            max_batch_size = batch_size

            # Clean up
            del model, input_tensor
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Try smaller batch size
                max_possible_batch_size = batch_size - 1

                # Clean up
                try:
                    del model, input_tensor
                    torch.cuda.empty_cache()
                except NameError:
                    pass
            else:
                raise e


