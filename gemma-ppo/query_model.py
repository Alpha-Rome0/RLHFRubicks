from transformers import AutoTokenizer
import torch

from trl import AutoModelForCausalLMWithValueHead
# login()

torch.set_default_device("cuda")

# phi-2
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# gemma-2b instruction tuned
model = AutoModelForCausalLMWithValueHead.from_pretrained("google/gemma-2b-it", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)

prompt = """
You are a Rubik's cube solving assistant. Your job is to generate the next best move 
when solving a Rubik's cube when given the a Rubik's cube scramble. A scramble is a list of 
moves that are performed on a fully solved Rubik's cube in order to scramble it up. When replying,
you must only reply with a single move.

Below I describe the possible moves:
(Up): Rotate the upper face 90 degrees clockwise.
U' (Up Prime): Rotate the upper face 90 degrees counter-clockwise.
U2: Rotate the upper face 180 degrees.
D (Down): Rotate the bottom face 90 degrees clockwise.
D' (Down Prime): Rotate the bottom face 90 degrees counter-clockwise.
D2: Rotate the bottom face 180 degrees.
F (Front): Rotate the front face 90 degrees clockwise.
F' (Front Prime): Rotate the front face 90 degrees counter-clockwise.
F2: Rotate the front face 180 degrees.
B (Back): Rotate the back face 90 degrees clockwise.
B' (Back Prime): Rotate the back face 90 degrees counter-clockwise.
B2: Rotate the back face 180 degrees.
L (Left): Rotate the left face 90 degrees clockwise.
L' (Left Prime): Rotate the left face 90 degrees counter-clockwise.
L2: Rotate the left face 180 degrees.
R (Right): Rotate the right face 90 degrees clockwise.
R' (Right Prime): Rotate the right face 90 degrees counter-clockwise.
R2: Rotate the right face 180 degrees.

Here is an example scramble and correct response.

Scramble: F2 B' U2 D' R2 L' U' B2 U2 B U' L2 U2 L U2 R B2 F2 R2 D
Response: R

Now you should generate the correct response for the following scramble (note: your answer should only contain a single move).
Scramble: D2 B2 R' D' R2 B' U2 L2 F' D2 L2 U2 B2 L U' B2 F U2 D' R'
Response:
"""
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda')

outputs = model.generate(**inputs, max_length=5000, pad_token_id=tokenizer.eos_token_id)

text = tokenizer.batch_decode(outputs)[0]

print(text)