import argparse
import logging
import sys
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig
from pynvml import *

import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
os.environ['PJRT_DEVICE'] ='GPU'

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def extract_data(data):
    # Initialize empty lists to store extracted data
    contexts = []     # Store contexts from paragraphs
    questions = []    # Store questions from Q&A pairs
    answers = []  # Store lists of answers corresponding to questions
    ids = []        # Store IDs of questions
    
    # Loop through the provided data
    for row in data:
        # Extract paragraphs within the data
        for paragraph in row['paragraphs']:
            # Extract Q&A pairs within each paragraph
            for qas in paragraph['qas']:
                # Store context for each Q&A pair
                contexts.append(paragraph['context'])
                
                # Store the question itself
                questions.append(qas['question'])
                
                # Store all answers related to the question
                answer_texts = []
                for answer in qas['answers']:
                    answer_texts.append(answer['text'])
                answers.append(answer_texts)
                
                # Store the ID of the question
                ids.append(qas['id'])
    
    # Return the extracted data
    return contexts, questions, answers, ids


def load_data(path):
    # Load JSON data from the given path
    obj = json.load(open(path))
    
    # Extract the 'data' field from the loaded JSON object
    obj = obj['data']
    
    # Process the extracted data to get context, question, answers, and IDs
    context, question, answers, ids = extract_data(obj)
    
    # Create a Dataset object using the processed data
    data = Dataset.from_dict({
        'context': context,
        'question': question,
        'answer': answers,
        'id': ids
    })
    
    # Return the constructed Dataset object
    return data

def format_samples(samples):
    prompted_text = []
    
    instruction = (
    "Act as a Multiple Answer Spans Healthcare Question Answering helpful assistant and answer the user's questions in details with reasoning. Do not give any false information. In case you don't have answer, specify why the question can't be answered.")
    
    q_header = "###QUESTION:"
    c_header = "###CONTEXT:"
    a_header = "###ANSWER:"
    
    for i in range(len(samples["question"])):
        q = samples['question'][i]
        c = samples["context"][i]
        a = ".".join([i.strip()+"\n" for i in samples["answer"][i]]) 
        text = f"{instruction}\n\n{q_header}\n{q}\n\n{c_header}\n{c}\n\n{a_header}\n{a}"
        prompted_text.append(text)
    
    return prompted_text


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="mistralai/Mistral-7B-Instruct-v0.1")
parser.add_argument("--train_size", type=float, default=0.80)
parser.add_argument("--sample", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lora_rank", type=int, default=64)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_bias", default="none")
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--eval_strategy", default="no")
parser.add_argument("--do_eval", default=False)
parser.add_argument("--eval_steps", type=int, default=20)
parser.add_argument("--train_batch_size", type=int, default=1)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--lr_scheduler_type", default="constant")
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--save_steps", type=int, default=20)
parser.add_argument("--train_in_bf16", action="store_true", default=True)
parser.add_argument("--output_dir", default="mlops-ft/")
parser.add_argument("--steps_dir", default="steps-ft/")
args = parser.parse_args()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.float16,
)

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    bias=args.lora_bias,
    lora_dropout=args.lora_dropout,
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj"

        ]
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path,
                                          add_eos_token=True)
tokenizer.add_special_tokens(special_tokens_dict={'pad_token': '[PAD]'})
tokenizer.padding_side = "right"

df_train = load_data("data/train_webmd_squad_v2_full.json")
df_train = df_train.shuffle(seed=42).select(range(100))

df_val = load_data("data/val_webmd_squad_v2_full.json")
df_val = df_val.shuffle(seed=42).select(range((30)))


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model_path, 
    quantization_config=bnb_config, 
    device_map="auto", 
)


# defining training arguments
training_args = TrainingArguments(
    output_dir=args.steps_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-4,
    bf16=False,
    tf32=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=format_samples,
    train_dataset=df_train,
    eval_dataset=df_train,
    peft_config=lora_config,
    max_seq_length=args.max_seq_length,
    packing=False,
)

result = trainer.train()
print_summary(result)
trainer.save_model(args.output_dir)

