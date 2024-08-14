from for_other_dataset_exp.llm4gnas.llms import *
from for_other_dataset_exp.llm4gnas.nas_method.nas_base import NASBase
from for_other_dataset_exp.llm4gnas.trainer import *
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.register import model_factory


from tqdm import tqdm
import torch
import logging


class GPT4GNAS(NASBase):
    def __init__(self, search_space: SearchSpaceBase, trainer: TrainerBase, config: dict, **kwargs):
        super().__init__(search_space, trainer, config)
        self.required_prompts = ["dataset", "connections", "N_operations", "operations"]
        self.dataset_name = self.config.dataset_name
        self.nas_iterations = self.config.nas_iterations if hasattr(self.config, "nas_iterations") else 20
        self.arch_dict = {}
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.build_llm()

    def check_response(self, response):
        return self.search_space.check_response(response)

    def check(self):
        for prompt in self.required_prompts:
            assert hasattr(self.search_space, prompt), f"Search Space {type(self.search_space)} hasn't offer the '{prompt}' prompt"

    def build_llm(self):
        llm = self.config.llm
        if llm == "ChatGPT":
            self.llm = ChatGPT(self.config.api_key, self.config.llm_model)

        elif llm == "ChatGPTProxy":
            self.llm = ChatGPTProxy(self.config.api_key, self.config.llm_model)
        else:
            raise RuntimeError(f"The LLM {llm} have not been achieve")

    def gen_prompt(self, stage):
        if stage == 0:
            return prompt_search_space(self.dataset_name, stage=stage, search_space=self.search_space)

        arch_dict = {}
        for key in self.arch_dict.keys():
            arch_dict[key] = self.arch_dict[key]

        return prompt_search_space(self.dataset_name, stage=stage, arch_dict=arch_dict, search_space=self.search_space)

    def gen_response(self, prompt):
        system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''
        history_chat = []

        if isinstance(self.llm, ChatGPT) or isinstance(self.llm, ChatGPTProxy):
            response = self.llm.response(system_content, prompt)
        return response

    def evaluate(self, archs, data, performance_history):
        if isinstance(self.search_space, Autogel_Space):
            gnn_list = [self.trainer.fit(data, self.search_space.to_gnn(desc=arch)) for arch in archs]
        else:
            gnn_list = [self.trainer.fit(data, self.search_space.to_gnn(arch.split(","))) for arch in archs]

        if self.config.parallel:
            trained_gnn_list = self.trainer.get_result(gnn_list)  # get the gnn after fit
        else:
            trained_gnn_list = gnn_list
        eval_var = "val acc"

        for arch, trained_gnn in zip(archs, trained_gnn_list):
            metric = self.trainer.evaluate(data, trained_gnn)
            self.arch_dict[arch] = metric[eval_var]
            performance_history.append({'arch': arch, 'trained_gnn': trained_gnn, 'metric': metric[eval_var]})


    def fit(self, data) -> GNNBase:
        performance_history = []
        for iteration in tqdm(range(self.nas_iterations)):
            prompt = self.gen_prompt(iteration)  # get prompt
            logging.info(prompt)
            response = self.gen_response(prompt)  # get llm's respone
            if response is None:
                raise RuntimeError("Error, please check the respone")
            archs = self.check_response(response)  # format response
            logging.info(archs)
            self.evaluate(archs, data, performance_history)
        # 获取最优架构
        logging.info(performance_history)
        best_performance = max(performance_history, key=lambda x: x.get('metric', 0))

        return best_performance['trained_gnn']

    def reset(self):
        self.llm = None
        self.best_model = None


model_factory["gpt4gnas"] = GPT4GNAS

####################################################################################################
# Prompts
####################################################################################################
def exp_prompt_nasgraph(arch_dict):
    prompt1 = '''\nHere are some experimental results that you can use as a reference:\n'''  
    prompt2 = '''\nThe model you propose should be strictly #different# from the structure of the existing experimental results.#You should not raise the models that are already present in the above experimental results again.#\n'''
    arch_l = list(arch_dict.keys())
    acc_l = [arch_dict[key] for key in arch_l]

    sorted_results = sorted(zip(arch_l, acc_l), key=lambda x: x[1], reverse=True)
    arch_l = [arch for arch, acc in sorted_results]
    acc_l = [acc for arch, acc in sorted_results]

    operation_repeat = set()
    operation_unique = []
    acc_unique = []
    seen = set()
    for i in range(len(arch_l)):
        if tuple(arch_l[i]) in seen:
            operation_repeat.add(arch_l[i])
        else:
            seen.add(tuple(arch_l[i]))
            operation_unique.append(arch_l[i])
            acc_unique.append(acc_l[i])

    prompt_repeat = ''''''
    if (len(operation_repeat) > 0):
        prompt_repeat = '''In the above experimental results, there are some repetitive models, as follows\n{}. #Please do not make the mistake of presenting the existing model in the experimental results again!#\n'''.format(
            ''.join(
                ['Model [{}]\n'.format(arch) for arch in operation_repeat]))

    prompt1 = prompt1 + '''{}#I hope you can learn the commonalities between the well performing models to achieve better results and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
        .format(''.join(
        ['Model [{}] achieves accuracy {:.4f} on the validation set.\n'.format(arch, acc) for arch, acc in
         zip(operation_unique, acc_unique)]))
    return prompt1 + prompt_repeat + prompt2


def prompt_search_space(dataname, arch_dict=None, stage=0, search_space=None):
    prompt1 = '''The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on {}, and the objective is to maximize model accuracy.'''.format(
        dataname)

    link_prompt = search_space.connections
    operation_prompt = search_space.operations
    prompt2 = '''
    Once again, your task is to help me find the optimal combination of operations corresponding to each layer of the model, while specifying the model structure and experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. We should select new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

    At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the operation space and identify which operation lists are promising. We can randomly select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.

    When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the operating space more effectively. We can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
        '''

    notice1 = '''\n#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#\n\n'''

    notice2 = '''\n#Due to the availability of a certain number of experimental results, I believe it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''

    suffix = search_space.example_gpt4gnas

    if (stage == 0):
        return prompt1 + link_prompt + operation_prompt + prompt2 + notice1 + suffix
    elif (stage < 4):
        return prompt1 + link_prompt + operation_prompt + prompt2 + exp_prompt_nasgraph(arch_dict) + notice1 + suffix
    else:
        return prompt1 + link_prompt + operation_prompt + prompt2 + exp_prompt_nasgraph(arch_dict) + notice2 + suffix

