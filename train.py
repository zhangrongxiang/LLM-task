from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# 将JSON文件转换为CSV文件
df = pd.read_json('./alpaca_data_cleaned.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained('./LLM-Research/gemma-2-2b-it')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<bos><start_of_turn>user\n{example['instruction'] + example['input']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


print(tokenizer.decode(tokenized_id[0]['input_ids']))


tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[3]["labels"])))

import torch

model = AutoModelForCausalLM.from_pretrained('./LLM-Research/gemma-2-2b-it', device_map="auto",low_cpu_mem_usage=True)

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
model.dtype

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config

model = get_peft_model(model, config)
model.print_trainable_parameters()


args = TrainingArguments(
    output_dir="./output/gemma-2-2b-3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_total_limit=5,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
with torch.cuda.amp.autocast():
    trainer.train()
#1925
trainer.model.config.save_pretrained("./my-model")

mode_path = './LLM-Research/gemma-2-2b-it'
lora_path = './output/gemma-2-2b-3/checkpoint-800' # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
from peft import PeftModel

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 调用模型进行对话生成
chat = [
    { "role": "user", "content": 'Hello' },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
outputs = tokenizer.decode(outputs[0])
response = outputs.split('model')[-1].replace('<end_of_turn>\n<eos>', '')
print(response)