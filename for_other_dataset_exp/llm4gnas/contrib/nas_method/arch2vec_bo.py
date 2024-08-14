import os
import sys
sys.path.insert(0, os.getcwd())
from llm4gnas.nas_method.nas_base import *
from llm4gnas.trainer.trainer_base import *
from llm4gnas.register import model_factory
import torch
import numpy as np
from collections import defaultdict
from torch.distributions import Normal
from llm4gnas.contrib.nas_method.arch2vec_model.model import Model
from llm4gnas.contrib.nas_method.arch2vec_model.configs import configs
from llm4gnas.contrib.nas_method.arch2vec_model.utils.utils import to_ops_nasbenchgraph, is_valid_nasbenchgraph, preprocessing
from llm4gnas.contrib.nas_method.pybnn.dngo import DNGO
from nas_bench_graph import light_read
from nas_bench_graph import Arch
import random
gnn_list = [
    "gat",  # GAT with 2 heads 0
    "gcn",  # GCN 1
    "gin",  # GIN 2
    "cheb",  # chebnet 3
    "sage",  # sage 4
    "arma",     # arma  5
    "graph",  # k-GNN 6
    "fc",  # fully-connected 7
    "skip"  # skip connection 8
]

link_list = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 0, 1, 3],
    [0, 1, 1, 1],
    [0, 1, 1, 2],
    [0, 1, 2, 2],
    [0, 1, 2, 3]
]


linkadj_dict = {
    '[0, 0, 0, 0]': 
    [[0, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],

    '[0, 0, 0, 1]':
    [[0, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],

    '[0, 0, 1, 1]':
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],

    '[0, 0, 1, 2]':
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],
     
    '[0, 0, 1, 3]':
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],

    '[0, 1, 1, 1]':
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],
     
    '[0, 1, 1, 2]':
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],

    '[0, 1, 2, 2]': 
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],],
     
    '[0, 1, 2, 3]': 
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
}


def _build_dataset(dataset):
    print(""" loading dataset """)
    nas_bench = light_read(dataset)
    X_adj = []
    X_ops = []
    labels = []
    transform_dict ={'input': 0, 'gat':1, 'gcn':2, 'gin':3, 'cheb':4, 'sage':5,
                 'arma':6, 'graph':7, 'fc':8, 'skip':9, 'output':10}
    for link in link_list:
        for i in gnn_list:
            for j in gnn_list:
                for k in gnn_list:
                    for l in gnn_list:
                        if i == 'skip' and j == 'skip' and k == 'skip' and l == 'skip':
                            continue
                        ops_one_hot = np.zeros([6, 11], dtype='int8')
                        ops_one_hot[0][transform_dict['input']]=1
                        ops_one_hot[1][transform_dict[i]]=1
                        ops_one_hot[2][transform_dict[j]]=1
                        ops_one_hot[3][transform_dict[k]]=1
                        ops_one_hot[4][transform_dict[l]]=1
                        ops_one_hot[5][transform_dict['output']]=1
                        arch = Arch(link, [i, j, k, l])
                        X_adj.append(torch.Tensor(linkadj_dict[f"{link}"]))
                        X_ops.append(torch.Tensor(ops_one_hot))
                        labels.append(torch.tensor([nas_bench[arch.valid_hash()]['perf']]))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    perf_labels = torch.stack(labels)
    return X_adj, X_ops, perf_labels


class Arch2vec_BO_NAS(NASBase):

    def __init__(
        self,
        search_space: SearchSpaceBase, 
        trainer: TrainerBase, 
        config: dict, 
        **kwargs
    ):
        super().__init__(search_space=search_space, trainer=trainer, config=config, **kwargs)
        self.config = config
        if self.config.isarch2vecbo == True:
            ### model config
            self.embed_dim = self.config.arch2vecbo.embed_dim
            self.input_dim = self.config.arch2vecbo.input_dim
            self.hidden_dim = self.config.arch2vecbo.hidden_dim
            self.num_hops = self.config.arch2vecbo.num_hops
            self.num_mlp_layers = self.config.arch2vecbo.num_mlp_layers
            self.dropout = self.config.arch2vecbo.dropout
            self.cfg = configs[self.config.arch2vecbo.arch2vec_cfg]
            ###
            self.num_epochs = self.config.epochs # search epoch
            self.dataset_name = self.config.dataset_name
            self.seed = self.config.arch2vecbo.seed
            self.init_size = self.config.arch2vecbo.init_size
            self.topk = self.config.arch2vecbo.topk
        else:
            print("The config \'isarch2vecbo\' is not True, please check!")
            exit()
        
        self.model = None

    def load_arch2vec(self):
        print("extract arch2vec on NAS_Bench_Graph search space ...")
        self.model = Model(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.embed_dim,
                    num_hops=self.num_hops, num_mlp_layers=self.num_mlp_layers, dropout=self.dropout, **self.cfg['GAE']).cuda()
        dir_name = './nas_method/arch2vec_model/pretrained/dim-{}'.format(self.embed_dim)
        if not os.path.exists(os.path.join(dir_name, 'model-nasbenchgraph.pt')):
            print("file not exist!")
            exit()
        self.model.load_state_dict(torch.load(os.path.join(dir_name, 'model-nasbenchgraph.pt').format(self.embed_dim))['model_state'])
        self.model.eval()
        X_adj, X_ops, perf_labels = _build_dataset(self.dataset_name)
        counter = 0
        embedding = {}
        for (adj, ops, labels) in zip(X_adj, X_ops, perf_labels):
            adj = torch.Tensor(adj).cuda()
            ops = torch.Tensor(ops).cuda()
            with torch.no_grad():
                adj, ops, prep_reverse = preprocessing(adj.unsqueeze(0), ops.unsqueeze(0), **self.cfg['prep'])
                x, _ = self.model._encoder(ops, adj)
                embedding[counter] = {'feature': x.mean(dim=1).squeeze(0).cpu(), 'perf': labels.squeeze(0).cpu(), 'origin_embedding': x.squeeze(0).cpu()}
            counter += 1
        random.seed(self.seed)
        random.shuffle(embedding)
        origin_embedding = [embedding[ind]['origin_embedding'] for ind in range(len(embedding))]
        features = [embedding[ind]['feature'] for ind in range(len(embedding))]
        perf_labels = [embedding[ind]['perf'] for ind in range(len(embedding))]
        features = torch.stack(features, dim=0).squeeze(0)
        perf_labels = torch.stack(perf_labels, dim=0).squeeze(0)
        origin_embedding = torch.stack(origin_embedding, dim=0).squeeze(0)
        return features, perf_labels, origin_embedding

    def get_init_samples(self, features, perf_labels, origin_embedding, visited):
        np.random.seed(self.seed)
        init_inds = np.random.permutation(list(range(features.shape[0])))[:self.init_size]
        init_inds = torch.tensor(init_inds).long()
        init_feat_samples = features[init_inds]
        init_perf_label_samples = perf_labels[init_inds]
        init_origin_embedding_samples = origin_embedding[init_inds]
        for idx in init_inds:
            visited[idx] = True
        return init_feat_samples, init_perf_label_samples, init_origin_embedding_samples, visited

    def propose_location(self, ei, features, perf_labels, origin_embedding, visited):
        k = self.topk
        print('remaining length of indices set:', len(features) - len(visited))
        indices = torch.argsort(ei)[-k:]
        ind_dedup = []
        for idx in indices:
            if idx not in visited:
                visited[idx] = True
                ind_dedup.append(idx)
        ind_dedup = torch.Tensor(ind_dedup).long()
        proposed_x, proposed_y, proposed_embed, = features[ind_dedup], perf_labels[ind_dedup], origin_embedding[ind_dedup]
        return proposed_x, proposed_y, proposed_embed, visited

    def is_valid_gnn(self, embed):
        op, ad = self.model.decoder(embed.unsqueeze(0).cuda())
        op = op.squeeze(0).cpu()
        ad = ad.squeeze(0).cpu()
        max_idx = torch.argmax(op, dim=-1)
        one_hot = torch.zeros_like(op)
        for i in range(one_hot.shape[0]):
            one_hot[i][max_idx[i]] = 1
        # decode the arch we find
        op_decode = to_ops_nasbenchgraph(max_idx)
        ad_decode = (ad>0.5).int().triu(1).numpy()
        ad_decode = np.ndarray.tolist(ad_decode)
        dist = [torch.dist(torch.Tensor(ad_decode), torch.Tensor(linkadj_dict[f"{link}"])) for link in link_list]
        link_decode = link_list[dist.index(min(dist))]
        is_valid = is_valid_nasbenchgraph(ad_decode, op_decode)
        return is_valid, op_decode, ad_decode, link_decode

    def expected_improvement_search(self, features, perf_labels, origin_embedding):
        """ implementation of expected improvement search given arch2vec.
        :return: features, labels
        """
        CURR_BEST_PERF = 0.
        CURR_BEST_OP = None
        CURR_BEST_LINK = None
        window_size = 32
        counter = 0
        epoch = 0.
        visited = {}
        best_trace = defaultdict(list)
        features, perf_labels, origin_embedding = features.cpu().detach(), perf_labels.cpu().detach(), origin_embedding.cpu().detach()
        feat_samples, perf_label_samples, origin_embedding_samples, visited = self.get_init_samples(features, perf_labels, origin_embedding, visited)
        for feat, perf, ori_embed in zip(feat_samples, perf_label_samples, origin_embedding_samples):
            counter += 1
            is_valid, op_decode, ad_decode, link_decode = self.is_valid_gnn(ori_embed)
            ### need to delete
            if "skip" in op_decode or "fc" in op_decode:
                continue
            ####
            if perf > CURR_BEST_PERF and is_valid:
                CURR_BEST_PERF = perf
                CURR_BEST_OP = op_decode
                CURR_BEST_LINK = link_decode
            best_trace['regret_perf'].append(float(1 - CURR_BEST_PERF))
            best_trace['counter'].append(counter)

        while epoch < self.num_epochs or CURR_BEST_OP is None:
            print("feat_samples:", len(feat_samples))
            print("current best perf: {}".format(CURR_BEST_PERF))
            model = DNGO(num_epochs=100, n_units=128, do_mcmc=False, normalize_output=False, rng=self.seed)
            model.train(X=feat_samples.numpy(), y=perf_label_samples.view(-1).numpy(), do_optimize=True)
            # print(model.network)
            m = []
            v = []
            chunks = int(features.shape[0] / window_size)
            if features.shape[0] % window_size > 0:
                chunks += 1
            features_split = torch.split(features, window_size, dim=0)
            for i in range(chunks):
                m_split, v_split = model.predict(features_split[i].numpy())
                m.extend(list(m_split))
                v.extend(list(v_split))
            mean = torch.Tensor(m)
            sigma = torch.Tensor(v)
            u = (mean - torch.Tensor([0.95]).expand_as(mean)) / sigma
            normal = Normal(torch.zeros_like(u), torch.ones_like(u))
            ucdf = normal.cdf(u)
            updf = torch.exp(normal.log_prob(u))
            ei = sigma * (updf + u * ucdf)
            feat_next, label_next_perf, ori_embed_next, visited = self.propose_location(ei, features, perf_labels, origin_embedding, visited)

            # add proposed networks to the pool

            for feat, perf, ori_embed in zip(feat_next, label_next_perf, ori_embed_next):
                is_valid, op_decode, ad_decode, link_decode = self.is_valid_gnn(ori_embed)
                ### need to delete
                if "skip" in op_decode or "fc" in op_decode:
                    continue
                ####
                if perf > CURR_BEST_PERF and is_valid:
                    CURR_BEST_PERF = perf
                    CURR_BEST_OP = op_decode
                    CURR_BEST_LINK = link_decode
                feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
                perf_label_samples = torch.cat((perf_label_samples.view(-1, 1), perf.view(1, 1)), dim=0)
                counter += 1
                best_trace['regret_perf'].append(float(1 - CURR_BEST_PERF))
                best_trace['counter'].append(counter)
            epoch += 1
            print("epoch: ", epoch)
            print("curr_best_perf: ", CURR_BEST_PERF)
            print("op: ", CURR_BEST_OP[1:-1])
            print("link: ", CURR_BEST_LINK)


        res = dict()
        res['regret_perf'] = best_trace['regret_perf']
        res['counter'] = best_trace['counter']
        print(CURR_BEST_PERF)
        print(CURR_BEST_OP)
        print(CURR_BEST_LINK)
        return self.search_space.to_gnn(CURR_BEST_OP[1:-1])

    def search(self):
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        features, perf_labels, origin_embedding = self.load_arch2vec()
        return self.expected_improvement_search(features, perf_labels, origin_embedding)
        
    
    def fit(self, data) -> GNNBase:
        self.best_model = self.search()
        return self.best_model

model_factory["arch2vec_bo_nas"] = Arch2vec_BO_NAS
