import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from reward import reward_model_basic, reward_model_strict
from dataset import load_data, RubiksDataset
from huggingface_hub import login
# login()


PHI_2_MODEL_ID = 'microsoft/phi-2'
GEMMA_MODEL_ID = 'google/gemma-2b-it'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# input arguments here
data_path = 'datasets/Kociemba_solutions.csv'
lr = 1e-4
batch_size = 1
epochs = 1


data = load_data(data_path)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.bos_token

# Load the model with 4bit quauntization
bnb_config = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        load_in_4bit=True
)
# model = AutoModelForCausalLM.from_pretrained(
#           GEMMA_MODEL_ID, 
#           quantization_config=bnb_config,
#           trust_remote_code=True
# )
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    GEMMA_MODEL_ID, 
    quantization_config=bnb_config,
    trust_remote_code=True
).to(device)

# model = prepare_model_for_kbit_training(model)

# Define the QLORA settings
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"] # these could change based on model
)

config = PPOConfig(
    model_name="gemma-2b-it",
    # learning_rate=lr,
    batch_size=batch_size,
    mini_batch_size=batch_size
)

dataset = RubiksDataset(tokenizer, data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(model)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
)

for epoch in tqdm(range(epochs), "epoch: "):
    for query_tensors, correct_answers in tqdm(dataloader): 
        query_tensors = query_tensors.squeeze(1)
        query_tensors = list(torch.unbind(query_tensors, dim=0))
        #### Get response from model
        response_tensors = [ppo_trainer.generate(query_tensor, max_length=570).squeeze(0) for query_tensor in query_tensors]
        responses = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        rewards = [torch.tensor(reward_model_strict(correct_answer, response), dtype=torch.float16) for correct_answer, response in zip(correct_answers, responses)]
    
        #### Run PPO stepda
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        #TODO: log stats
        # ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model(f"gemma-2b-it-rlhf-kociemba")