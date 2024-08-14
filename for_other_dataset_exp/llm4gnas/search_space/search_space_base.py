import re
from easydict import EasyDict as edict
from typing import List, Dict, Union
from for_other_dataset_exp.llm4gnas.register import model_factory
from for_other_dataset_exp.llm4gnas.search_space.gnn_base import *
from for_other_dataset_exp.llm4gnas.search_space.autogel_space import *


class SearchSpaceBase(object):
    def __init__(self, config, **kwargs):
        self.config = config

    def get_operations(self) -> Union[List, Dict]:
        raise NotImplementedError

    def to_gnn(self, desc: Union[List, str]) -> GNNBase:
        raise NotImplementedError




class Autogel_Space(SearchSpaceBase):
    # prompts of this search space
    connections = """ The definition of search space for models follow the neighborhood aggregation schema, i.e., the Message Passing Neural Network (MPNN), which is formulated as:
{
    $$\mathbf{m}_{v}^{k+1}=AGG_{k}(\{M_{k}(\mathbf{h}_{v}^{k}\mathbf{h}_{u}^{k},\mathbf{e}_{vu}^{k}):u\in N(v)\})$$
    $$\mathbf{h}_{v}^{k+1}=ACT_{k}(COM_{k}(\{\mathbf{h}_{v}^{k},\mathbf{m}_{v}^{k+1}\}))$$
    $$\hat{\mathbf{y}}=R(\{\mathbf{h}_{v}^{L}|v\in G\})$$
    where $k$ denotes $k$-th layer, $N(v)$ denotes a set of neighboring nodes of $v$, $\mathbf{h}_{v}^{k}$, $\mathbf{h}_{u}^{k}$ denotes hidden embeddings for $v$ and $u$ respectively, $\mathrm{e}_{vu}^{k}$ denotes features for edge e(v, u) (optional), $\mathbf{m}_{v}^{k+1}$denotes the intermediate embeddings gathered from neighborhood $N(v)$, $M_k$ denotes the message function, $AGG_{k}$ denotes the neighborhood aggregation function, $COM_{k}$ denotes the combination function between intermediate embeddings and embeddings of node $v$ itself from the last layer, $ACT_{k}$ denotes activation function. Such message passing phase in repeats for $L$ times (i.e.,$ k\in\{1,\cdots,L\}$). For graph-level tasks, it further follows the readout phase in where information from the entire graph $G$ is aggregated through readout function $R(·)$.
}"""
    N_operations = """ {
    agg: [sum,mean,max];
    combine: [sum,concat];
    act: [relu,prelu];
    layer_connect: [stack,skip_sum,skip_cat];
    layer_agg: [none,concat,max_pooling];
}"""
    operations = """ The parts that need to be determined in the search space through architecture search, along with their corresponding selectable functions, are as follows:
{
    agg: [sum,mean,max];
    combine: [sum,concat];
    act: [relu,prelu];
    layer_connect: [stack,skip_sum,skip_cat];
    layer_agg: [none,concat,max_pooling];
}
#Define the model as a two-layer GNN model, where you need to choose functions agg, combine, act, and layer_connect for each layer, and also need to select a layer_agg function for the entire model.#
"""
    example_gpt4gnas = """ Please answer me as briefly as possible. You should give 10 different model structures at a time.
    For example:
    1.Model:{layer1:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_cat}; layer2:{ agg:sum, combine:concat, act:relu,layer_connect:stack}; layer_agg:concat;}
    2.Model:{layer1:{ agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer2:{ agg:max, combine:sum, act:prelu, layer_connect:skip_cat}; layer_agg:max_pooling;}
    3.Model: ...
    ...
    10.Model:{layer1:{ agg:max, combine:sum, act:relu, layer_connect:stack}; layer2:{agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer_agg:concat;}
    And The response you give must strictly follow the format of this example. """

    def get_operations(self):
        openration_dirt = {
            'agg': ['sum', 'mean', 'max'],
            'combine': ['sum', 'concat'],
            'act': ['relu', 'prelu'],
            'layer_connect': ['stack', 'skip_sum', 'skip_cat'],
            'layer_agg': ['none', 'concat', 'max_pooling'],
            'pool': ['global_add_pool', 'global_mean_pool', 'global_max_pool']
        }
        if self.config.taskname == "LinkPredict":
            openration_dirt['pool'] = ['sum', 'max', 'concat']
        return openration_dirt

    def to_gnn(self, desc="{layer1:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_cat}; layer2:{ agg:sum, combine:concat, act:relu,layer_connect:stack}; layer_agg:concat;}"):
        return Autogel(desc=desc, config=self.config)

    def check_response(self, response):
        input_lst = response.split('Model:')[1:]
        archs = []
        for string in input_lst:
            new_string = re.sub(r'\d+\.\s*$', '', string)
            archs.append(new_string)
        return archs


model_factory["autogel_space"] = Autogel_Space

