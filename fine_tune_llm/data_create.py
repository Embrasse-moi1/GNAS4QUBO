from nas_bench_graph import light_read, Arch
import json


# 根据架构空间选择对应的架构连接方式
def arch_to_dict(arch):
    struct_dict = {
        (0, 0, 0, 0): "[[0, 1, 1, 1, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 0, 0, 1): "[[0, 1, 1, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 0, 1, 1): "[[0, 1, 1, 0, 0, 0],[0, 0, 0, 1, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 0, 1, 2): "[[0, 1, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 0, 1, 3): "[[0, 1, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 1, 1, 1): "[[0, 1, 0, 0, 0, 0],[0, 0, 1, 1, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 1, 1, 2): "[[0, 1, 0, 0, 0, 0],[0, 0, 1, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 1, 2, 2): "[[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]",
        (0, 1, 2, 3): "[[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0]]"
    }
    arch_tuple = tuple(arch)
    return struct_dict.get(arch_tuple)


# 根据架构空间获得性能性能较高的架构内容
def arch_performance(data, arch_list):
    bench = light_read(data)
    operation = ['gcn', 'gat', 'sage', 'gin', 'cheb', 'arma', 'graph', 'fc', 'skip']
    operation_pre = {}
    operation_valid_perf= {}
    for operation_one in operation:
        for operation_two in operation:
            for operation_three in operation:
                for operation_four in operation:
                    arch = Arch(arch_list, [operation_one, operation_two, operation_three, operation_four])
                    operation_tuple = (operation_one,operation_two,operation_three,operation_four)
                    try:
                        info = bench[arch.valid_hash()]
                    except KeyError:
                        continue
                    else:
                        operation_pre[operation_tuple] = round(info['perf'], 4)
                        operation_valid_perf[operation_tuple] = round(info['valid_perf'], 4)
    # 排名前三的架构性能（测试集）
    operation_pre_sorted = sorted(operation_pre.items(), key=lambda x: x[1], reverse=True)
    operation_pre_list = operation_pre_sorted[:3]
    operation_pre_dict = {k: v for k, v in operation_pre_list}

    # 排名前三的架构性能（验证集）
    operation_valid_pre_sorted = sorted(operation_valid_perf.items(), key=lambda x: x[1], reverse=True)
    operation_valid_pre_list = operation_valid_pre_sorted[:3]
    operation_valid_pre_dict = {k: v for k, v in operation_valid_pre_list}

    return operation_pre_dict


# 构建微调数据的指令、输入和输出
def create_data(data, arch):
    struct_dict = arch_to_dict(arch)
    arch_pre_dict = arch_performance(data, arch)
    instruction = f'''The task is to provide some helpful graph neural network architectures based on a given dataset. \
These architectures will be trained and tested on {data}, and the architectures you provide should enable the model to achieve high accuracy.\n\
The connection method of the architecture is as follows: The first operation is the input, the last operation is the output,\
and the middle operations are candidate operations. The adjacency matrix for the operation connections is as follows:{struct_dict}, \
where the element (i,j) in the adjacency matrix indicates that the output of operation i will be used as the input for operation j.\n\
There are nine candidate operations for the architecture: {{gcn, gat, sage, gin, cheb, arma, graph, fc, skip}}.'''
    input = "Please return some architecture models based on the GNN architecture and the relevant dataset I provided. Each model should contain four operations."
    output = f'''Here are some architectures that perform well on the test dataset and their corresponding test accuracies:{arch_pre_dict}'''
    return instruction, input, output


# 创建微调数据集并存储在json文件中
def create_finetune_data(data_options, struct_dict_arch):
    finetune_data = []
    for data_select in data_options:
        for struct_arch in struct_dict_arch:
            instruction, input, output = create_data(data_select, struct_arch)
            # 将所有数据都存放到列表中
            data_dictionary = {
                "instruction": instruction,
                "input": input,
                "output": output
            }
            finetune_data.append(data_dictionary)
    data_str = json.dumps(finetune_data, ensure_ascii=False, indent=4)
    with open('finetune.json', 'w', encoding='utf-8') as file:
        file.write(data_str)
    file.close()


if __name__ == '__main__':
    data_options = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']
    struct_dict_arch = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3], [0, 1, 1, 1],
                        [0, 1, 1, 2], [0, 1, 2, 2], [0, 1, 2, 3]]
    create_finetune_data(data_options,struct_dict_arch)
