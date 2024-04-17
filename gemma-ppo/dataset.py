import torch
from torch.utils.data import Dataset, DataLoader
import csv


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def get_prompt(scramble):

    return f"""You are a Rubik's cube solving assistant. Your job is to generate the next best move when solving a Rubik's cube when given the a Rubik's cube scramble. A scramble is a list of moves that are performed on a fully solved Rubik's cube in order to scramble it up. When replying, you must only reply with a single move.

    Below are the only valid next moves:
    U (Up): Rotate the upper face 90 degrees clockwise.
    U' (Up Prime): Rotate the upper face 90 degrees counter-clockwise.
    U2 (Up twice): Rotate the upper face 180 degrees.
    D (Down): Rotate the bottom face 90 degrees clockwise.
    D' (Down Prime): Rotate the bottom face 90 degrees counter-clockwise.
    D2 (Down twice): Rotate the bottom face 180 degrees.
    F (Front): Rotate the front face 90 degrees clockwise.
    F' (Front Prime): Rotate the front face 90 degrees counter-clockwise.
    F2 (Front twice): Rotate the front face 180 degrees.
    B (Back): Rotate the back face 90 degrees clockwise.
    B' (Back Prime): Rotate the back face 90 degrees counter-clockwise.
    B2 (Back twice): Rotate the back face 180 degrees.
    L (Left): Rotate the left face 90 degrees clockwise.
    L' (Left Prime): Rotate the left face 90 degrees counter-clockwise.
    L2 (Left twice): Rotate the left face 180 degrees.
    R (Right): Rotate the right face 90 degrees clockwise.
    R' (Right Prime): Rotate the right face 90 degrees counter-clockwise.
    R2 (Right twice): Rotate the right face 180 degrees.

    Here is an example scramble and correct response.

    Scramble: F2 B' U2 D' R2 L' U' B2 U2 B U' L2 U2 L U2 R B2 F2 R2 D
    Next move: R

    Now you should generate the correct next move for the following scramble (note: your answer should only contain a single move and nothing more).
    Scramble: {scramble}
    Next move: """

def load_data(data_path, num_predicted_turns=10000):
    with open(data_path, 'r') as data:
        # Using the csv library to read the data
        reader = csv.DictReader(data)

        # Creating the list of dictionaries
        result = []
        for row in reader:
            result.append({
                'query': row['Scramble (State)'].strip(),
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