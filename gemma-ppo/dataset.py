import torch
from torch.utils.data import Dataset, DataLoader
import csv


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
def get_prompt(scramble):

    return f"""You are a 2x2 Rubik's cube solving assistant. Your job is to generate the space separated moves needed to solve a Rubik's cube when given the Rubik's cube scramble. A scramble is a list of moves that are performed on a fully solved Rubik's cube in order to scramble it up.

    Below are the only valid next moves:

    U
    U'
    U2
    F
    F'
    F2
    R
    R'
    R2

    Here is an example scramble and correct response.

    Scramble: F
    Next moves: F'

    Now you should generate the correct next move to solve the 2x2 rubiks cube following the scramble. (note: your answer should only contain one move).
    Scramble: {scramble}
    Next moves: """

def load_data(data_path, num_predicted_turns=10000):
    with open(data_path, 'r') as data:
        # Using the csv library to read the data
        reader = csv.DictReader(data)

        # Creating the list of dictionaries
        result = []
        for row in reader:
            result.append({
                'query': row['Scramble'].strip(),
                'output': ' '.join(row['Solution'].strip().split(' ')[:num_predicted_turns])
            })

    return result


class RubiksDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query_encoding = self.tokenizer(get_prompt(item['query']), return_tensors='pt', padding='max_length', max_length=550)
        # response_encoding = self.tokenizer(item['response'], return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
        return query_encoding['input_ids'].to(device), item['query'], item['output']