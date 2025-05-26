import tiktoken
import torch
import os
import time
from c2_bonus_raschka_bpe_openai_gpt2 import get_encoder, download_vocab
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from c2_bonus_raschka_bpe_from_scratch import BPETokenizerSimple
import timeit

'''
Vergleich von verschiedenen BPE (Byte-Pair-Encoding) Implementierungen 
- von tiktoken
- von GPT-2
- von Hugging Face Transformer
'''



def main():

    ### BPE von Tiktoken ###
    start_time = time.time()
    tik_tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world. Is this-- a test?"

    integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print (integers)

    strings = tik_tokenizer.decode(integers)
    print(strings)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"BPE Tiktoken completed in {execution_time_minutes:.2f} seconds.\n\n")


    ### Original BPE Implementierung, die in GPT-2 genutzt wird ###
    #download_vocab()
    start_time2 = time.time()
    orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
    integers = orig_tokenizer.encode(text)
    print(integers)

    strings = orig_tokenizer.decode(integers)
    print(strings)
    end_time2 = time.time()
    execution_time_minutes = (end_time2 - start_time2) 
    print(f"BPE GPT-2 completed in {execution_time_minutes:.2f} seconds.\n\n")


    ### BPE mit Hugging Face transformers ###
    start_time3 = time.time()
    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(hf_tokenizer(strings)["input_ids"])

    hf_tokenizer_fast = GPT2TokenizerFast.from_pretrained("gpt2")
    print(hf_tokenizer_fast(strings)["input_ids"])
    end_time3 = time.time()
    execution_time_minutes = (end_time3 - start_time3) 
    print(f"BPE Hugging Face completed in {execution_time_minutes:.2f} seconds.\n\n")


    ### Raschka from-scratch BPE Tokenizer ###
    start_time4 = time.time()
    tokenizer_gpt2 = BPETokenizerSimple()
    
    tokenizer_gpt2.load_vocab_and_merges_from_openai(
    vocab_path=os.path.join("gpt2_model", "encoder.json"),
    bpe_merges_path=os.path.join("gpt2_model", "vocab.bpe")
    )
    integers = tokenizer_gpt2.encode(text)
    print(integers)
    end_time4 = time.time()
    execution_time_minutes = (end_time4 - start_time4) 
    print(f"custome BPE Raschka completed in {execution_time_minutes:.2f} seconds.\n\n")


    ### PERFORMENCE BENCHMAK ###
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Original OpenAI GPT-2 Tokenizer
    start_time = time.time()
    orig_tokenizer.encode(raw_text)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Original OpenAI GPT-2 Tokenizer completed in {execution_time_minutes:.2f} seconds.\n\n")


    #Tiktoken OpenAI Gpt-2 Tokenizer
    start_time = time.time()
    tik_tokenizer.encode(raw_text)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Tiktoken OpenAI Gpt-2 Tokenizer completed in {execution_time_minutes:.2f} seconds.\n\n")

    # Hugging Face OpenAI Gpt-2 Tokenizer
    start_time = time.time()
    hf_tokenizer(raw_text)["input_ids"]
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Hugging Face OpenAI Gpt-2 Tokenizer 1 completed in {execution_time_minutes:.2f} seconds.\n\n")

    start_time = time.time()
    hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Hugging Face OpenAI Gpt-2 Tokenizer 2 completed in {execution_time_minutes:.2f} seconds.\n\n")

    start_time = time.time()
    hf_tokenizer_fast(raw_text)["input_ids"]
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Hugging Face OpenAI Gpt-2 Tokenizer 3 completed in {execution_time_minutes:.2f} seconds.\n\n")

    start_time = time.time()
    hf_tokenizer_fast(raw_text, max_length=5145, truncation=True)["input_ids"]
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Hugging Face OpenAI Gpt-2 Tokenizer 4 completed in {execution_time_minutes:.2f} seconds.\n\n")

    # Raschka GPT-2 Tokenizer
    start_time = time.time()
    tokenizer_gpt2.encode(raw_text)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) 
    print(f"Raschka GPT-2 Tokenizer completed in {execution_time_minutes:.2f} seconds.\n\n")






if __name__ == "__main__":
    main()