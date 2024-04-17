import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from reward import reward_model_basic, reward_model_strict, reward_model_distance
from dataset import load_data, RubiksDataset
from huggingface_hub import login
import bitsandbytes as bnb

from pycubescrambler import nxn,side,non

# Python >3.9 compatibility
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from rubik_solver.Cubie import Cube
from rubik_solver import utils
from rubik_solver.Move import Move

# Plot histogram
import matplotlib.pyplot as plt
import numpy as np
# login()

device = "cuda:0"

LR = 1.41e-5
BATCH_SIZE = 1
MINI_BATCH_SIZE = 1

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.float16
)

model_id = "google/gemma-2b-it"
GEMMA_MODEL_ID = 'google/gemma-2b-it'
#model_id = "vicgalle/gpt2-open-instruct-v1"
config = PPOConfig(
    model_name=model_id,
    learning_rate=LR,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name, quantization_config=bnb_config, peft_config=lora_config, device_map="auto")
#model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name,  device_map="auto", token=os.environ['HF_TOKEN'])

optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LR)

data_path = 'datasets/Kociemba_solutions.csv'
data = load_data(data_path)
dataset = RubiksDataset(tokenizer, data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
    optimizer= optimizer
)

epochs = 1

for epoch in tqdm(range(epochs), "epoch: "):
    for query_tensors, query, correct_answers in tqdm(dataloader): 
        query_tensors = query_tensors.squeeze(1)
        query_tensors = list(torch.unbind(query_tensors, dim=0))
        #### Get response from model
        response_tensors = [ppo_trainer.generate(query_tensor, max_length=570).squeeze(0) for query_tensor in query_tensors]
        responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        cube = Cube()
        for move in query[0].split(' '):
            cube.move(Move(move))
        #### Compute reward score
        rewards = [torch.tensor(reward_model_distance(cube, correct_answer, response, 'Kociemba'), dtype=torch.float16) for correct_answer, response in zip(correct_answers, responses)]
    
        #### Run PPO stepda
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        #TODO: log stats
        # ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model(f"gemma-2b-it-rlhf-kociemba")