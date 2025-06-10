import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import GPT2LMHeadModel, pipeline, set_seed
import matplotlib.pyplot as plt
import tiktoken


#--------------------------------------------------#

### Testen und Ausführen ####
def main():

    # Laden vom Model; model_hf steht für model huggingface
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # Das ist das 124M Model
    
    # sd_hf => state_dict hugging face
    sd_hf = model_hf.state_dict() # hol die Raw Tensoren aus dem GPT2 Model

    # Ausdruck der Werte
    for k, v in sd_hf.items():
        print(k, v.shape) # k = key, v = values


    # Beispiel
    print(sd_hf["transformer.wpe.weight"].view(-1)[:20])

    # Grafische Darstellung der feste Positionen 
    # Das gezeigte graue Fenster weißt Strukturen auf und jede Zeile 
    # ist die Visualisierung von einer anderen Positon > einer festen absoluten Position
    # Range 0 - 1024
    
    '''
    # Ein Gesamtbild
    plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
    
    # Hier schaut man sich einzelne Spalten an 
    plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
    plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
    plt.plot(sd_hf["transformer.wpe.weight"][:, 250])

    # wie sieht die grafische Darstellung von einem Head aus mit attention weights
    plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300, :300], cmap="gray")

    plt.show()
    '''

    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    print(generator("Hello, I´m a language model,", max_length=30, num_return_sequences=5))
    

    # Shakespear Text (Beispieltext)
    with open('BHT_BA2025_LLM_Karpathy/input.txt', 'r') as file:
        text = file.read()
    
    data = text[:1000] # die ersten 1000 Zeichen
    print(data[:100])

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(data)
    print(tokens[:24])










if __name__ == "__main__":
    main()