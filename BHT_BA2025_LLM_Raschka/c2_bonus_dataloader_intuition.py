import torch
from torch.utils.data import DataLoader, Dataset

'''
Zum besseren Verständnis des Dataloaders, welcher die Sliding Window Attention nutzt. 
'''

### Klassen ###

# weicht von der GPTDatasetV1 in der Datei c2_data_preparation_sampling ab
class GPTDatasetV1a(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Modification
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


### Funtkionen ###

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None

    # Create dataset
    dataset = GPTDatasetV1a(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



### Teste + Ausführung in Main() ###
def main():

    with open("number-data.txt", "w", encoding="utf-8") as f:
        for number in range(1001):
            f.write(f"{number} ")


    with open("number-data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Erster Batch
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    # Zweiter Batch
    second_batch = next(data_iter)
    print(second_batch)

    # Dritter Batch
    third_batch = next(data_iter)
    print(third_batch)

    # Nun für alle Batch Inputs + Mischung der Startwerte
    dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

    for inputs, targets in dataloader:
        pass

    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


    # Dataloader mit Random Startwerten
    torch.manual_seed(123)
    dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

    for inputs, targets in dataloader:
        pass

    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)







if __name__ == "__main__":
    main()