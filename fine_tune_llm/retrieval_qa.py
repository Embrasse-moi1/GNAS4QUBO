from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType


def pre_process(model_path, lora_path):
    # 加载tokenizer，并调整词嵌入
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 加载模型并调整词汇嵌入大小
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    # 加载LoRA权重
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
    return tokenizer, model


def llama3(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,  # 确保传递attention_mask
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.01,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0],
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


if __name__ == '__main__':
    model_path = '/home/zhangzg/LLM/model/Llama-3-8B-Instruct'
    lora_path = './llama3_finetune'
    tokenizer, model = pre_process(model_path, lora_path)
    prompt1 = '''The task is to provide some helpful graph neural network architectures based on a given dataset. \
These architectures will be trained and tested on cora, and the architectures you provide should enable the model to achieve high accuracy.\n\
The connection method of the architecture is as follows: The first operation is the input, the last operation is the output,\
and the middle operations are candidate operations. The adjacency matrix for the operation connections is as follows:[[0, 1, 1, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]], \
where the element (i,j) in the adjacency matrix indicates that the output of operation i will be used as the input for operation j.\n\
There are nine candidate operations for the architecture: {{gcn, gat, sage, gin, cheb, arma, graph, fc, skip}}.\n\
Please return some architecture models based on the GNN architecture and the relevant dataset I provided. Each model should contain four operations.'''
    response1 = llama3(prompt1, model, tokenizer)
    print(response1)
    print('='*100)
    # prompt2 = "What is the capital of China?"
    # response2 = llama3(prompt2, model, tokenizer)
    # print(response2)
