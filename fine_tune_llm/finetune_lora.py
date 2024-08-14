import torch
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model


# 数据预处理
df = pd.read_json('./finetune.json')
ds = Dataset.from_pandas(df)
print(ds[:3])  # 打印数据集的前3条记录

tokenizer = AutoTokenizer.from_pretrained('/home/zhangzg/LLM/model/Llama-3-8B-Instruct', use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False)
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
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


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_ds)
print(tokenizer.decode(tokenized_ds[0]['input_ids']))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

# 模型加载与配置
model = AutoModelForCausalLM.from_pretrained('/home/zhangzg/LLM/model/Llama-3-8B-Instruct',
                                             device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
print(model.dtype)

# 应用LoRA技术
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.config.use_cache = False
print(model.print_trainable_parameters())

# 训练参数设置
args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=50,
    num_train_epochs=100,
    save_steps=50,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True
)

# 模型训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,  # 确保这里传入的是处理好的数据集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

# 模型保存
peft_model_id = "./llama3_finetune"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
