from untils import main_prompt_word
import re
import openai
from train_gnn import *
import json
import requests

# Get GNN architecture from GPT
openai.api_key = "sb-55a412fbe6958d0982e73f8765bb7dfaf38c254c4c5b6b67"
openai.api_base = "https://api.openai-sb.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + openai.api_key
}

gnn_list = [
    "gat",  # GAT with 2 heads 0
    "gcn",  # GCN 1
    "gin",  # GIN 2
    "cheb",  # chebnet 3
    "sage",  # sage 4
    "arma",  # 5
    "graph",  # k-GNN 6
    "fc",  # fully-connected 7
    "skip"  # skip connection 8
]

link_list = [
    # [0, 0, 0, 0],
    # [0, 0, 0, 1],
    # [0, 0, 1, 1],
    # [0, 0, 1, 2],
    # [0, 0, 1, 3],
    [0, 1, 1, 1],
    [0, 1, 1, 2],
    [0, 1, 2, 2],
    # [0, 1, 2, 3]
]
operation_dict = {'GCN': 'gcn', 'GAT': 'gat', 'GraphSAGE': 'sage', 'GIN': 'gin', 'ChebNet': 'cheb', 'ARMA': 'arma',
                  'k-GNN': 'graph', 'skip': 'skip', 'fully-connected-layer': 'fc'}
dataname = 'Gset published by stanford university which is related to graph theory, and the Gset dataset are random d-regular graphs'
system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''
# link have 9 chioces

if __name__ == "__main__":
    for link in link_list:
        # 写入GNN宏观架构
        with open("experiment.txt", "a") as file:
            file.write(str(link) + "\n")
        messages = [{"role": "system", "content": system_content},
                    {"role": "user", "content": main_prompt_word(link=tuple(link), dataname=dataname, stage=0)}, ]
        payload = {
            "model": 'gpt-4',
            "messages": messages,
            "temperature": 0
        }
        all_egdes = 4694  # the sum of egdes of G15
        arch_list = []
        # acc_list = []
        messages_history = []
        iterations = 10

        for iteration in range(iterations):
            with open("experiment.txt", "a") as file:
                file.write("Epoch" + str(iteration) + "\n")
            print(iteration)
            option_list = []

            try:
                response = requests.post(openai.api_base, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                res = response.json()
                result_value = res['choices'][0]['message']['content']
                print(result_value)
            except (requests.HTTPError, json.JSONDecodeError) as err:
                print("JSON parsing error:", err)
            except Exception as err:
                print("Other exceptions:", err)

            messages.append(res)  # 直接在传入参数 messages 中追加消息
            messages_history.append(messages)
            # res_temp = res['content']
            input_lst = re.split('Model:|model:', result_value)

            for i in range(1, len(input_lst)):
                operations_str = input_lst[i].split('[')[1].split(']')[0]
                operations_list = operations_str.split(',')
                # ['gcn', ' gat', ' sage', ' gin']
                operations_list_str = [a.replace(" ", "") for a in operations_list]  # 获得去除了空格的列表

                option_list.append(operations_list_str)
                print(operations_list_str)
                arch_list.append({'arch_Operations': operations_str})

            acc_list = get_acc_list(link, all_egdes, option_list)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",
                 "content": main_prompt_word(link=tuple(link), dataname=dataname, arch_list=arch_list,
                                             acc_list=acc_list,
                                             stage=iteration)},
            ]
            print(messages)


