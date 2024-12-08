import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from options import parse_args
# Load the TruthfulQA dataset
truthfulqa_path = 'alpaca_data_cleaned.json'
df_truthfulqa = pd.read_json(truthfulqa_path)
ds_truthfulqa = Dataset.from_pandas(df_truthfulqa)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('./LLM-Research/gemma-2-2b-it')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

# Load Gemma model
model = AutoModelForCausalLM.from_pretrained('./LLM-Research/gemma-2-2b-it', device_map="auto", low_cpu_mem_usage=True)
print(model)


# Process function (same as you used for Alpaca dataset)
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<bos><start_of_turn>user\n{example['instruction'] + example['input']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Tokenize the TruthfulQA dataset
tokenized_id = ds_truthfulqa.map(process_func, remove_columns=ds_truthfulqa.column_names)

def freeze_column(model,col,index):
    # Access the transformer layers stored in model.model.layers
    q_proj=model.model.model.layers[index].self_attn.q_proj
    weight=q_proj.weight
    print(f"The Q matrix has {weight.shape[1]} columns to be freezed.")
    weight[:,col].requires_grad=False
    print(f'The {col} column is freezed.\n')


# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Load LoRA configuration for fine-tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False,
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

print(model.model)
print(model.model.model)

def train_and_save(path,layers,model):
    # Define training arguments
    args = TrainingArguments(
        output_dir=path,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_total_limit=5,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        logging_dir='./logs',  # Log directory for monitoring
    )
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # Train the model
    with torch.cuda.amp.autocast():
        trainer.train()
from options import parse_args
args=parse_args()
col=args.col
index=args.index
import torch

freeze_column(model,col=col,index=index)
training_time, eval_results = train_and_save(f"./output/task3_{index}_{col}",col,model)
print(f"Training time: {training_time:.2f} seconds")


print(f"Evaluation results: {eval_results}")

