import os
import sys
sys.path.insert(0, os.getcwd())
from llm4gnas.nas_method.nas_base import *
from llm4gnas.contrib.nas_method.arch2vec_model import Model
from llm4gnas.contrib.nas_method.arch2vec_model.configs import configs
from llm4gnas.contrib.nas_method.arch2vec_model.utils.utils import to_ops_nasbenchgraph, is_valid_nasbenchgraph, preprocessing
from llm4gnas.trainer.trainer_base import *
from nas_bench_graph import light_read
from nas_bench_graph import Arch
from llm4gnas.register import model_factory
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

'''
build dataset for arch2vec_rl_nas to get embeddings
'''
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

class Env(object):
    def __init__(self, name, embed_dim, input_dim, hidden_dim, num_hops, num_mlp_layers, dropout, cfg):
        self.name = name
        self.dir_name = None
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        self.visited = {}
        self.features = []
        self.perf_labels = []
        self.embedding = {}
        self._reset(cfg)

    def _reset(self, cfg):
        print("extract arch2vec on NAS_Bench_Graph search space ...")
        self.model = Model(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.embed_dim,
                    num_hops=self.num_hops, num_mlp_layers=self.num_mlp_layers, dropout=self.dropout, **cfg['GAE']).cuda()
        self.dir_name = './nas_method/arch2vec_model/pretrained/dim-{}'.format(self.embed_dim)
        if not os.path.exists(os.path.join(self.dir_name, 'model-nasbenchgraph.pt')):
            print("file not exist!")
            exit()
        self.model.load_state_dict(torch.load(os.path.join(self.dir_name, 'model-nasbenchgraph.pt').format(self.embed_dim))['model_state'])
        self.model.eval()
        X_adj, X_ops, perf_labels = _build_dataset(self.name)
        counter = 0
        for (adj, ops, labels) in zip(X_adj, X_ops, perf_labels):
            adj = torch.Tensor(adj).cuda()
            ops = torch.Tensor(ops).cuda()
            with torch.no_grad():
                adj, ops, prep_reverse = preprocessing(adj.unsqueeze(0), ops.unsqueeze(0), **cfg['prep'])
                x, _ = self.model._encoder(ops, adj)
                self.embedding[counter] = {'feature': x.mean(dim=1).squeeze(0).cpu(), 'origin_embedding': x.squeeze(0).cpu(),'perf': labels.squeeze(0).cpu()}
            counter += 1
        random.shuffle(self.embedding)
        self.features = [self.embedding[ind]['feature'] for ind in range(len(self.embedding))]
        self.perf_labels = [self.embedding[ind]['perf'] for ind in range(len(self.embedding))]
        self.origin_embedding = [self.embedding[ind]['origin_embedding'] for ind in range(len(self.embedding))]
        self.features = torch.stack(self.features, dim=0).squeeze(0)
        self.perf_labels = torch.stack(self.perf_labels, dim=0).squeeze(0)
        self.origin_embedding = torch.stack(self.origin_embedding, dim=0).squeeze(0)
        print("finished arch2vec extraction")


    def get_init_state(self):
        """
        :return: 1 x dim
        """
        rand_indices = random.randint(0, self.features.shape[0]-1)
        self.visited[rand_indices] = True
        return self.features[rand_indices], self.perf_labels[rand_indices], self.origin_embedding[rand_indices]
    def step(self, action):
        """
        action: 1 x dim
        self.features. N x dim
        """
        dist = torch.norm(self.features - action.cpu(), dim=1)
        knn = (-1 * dist).topk(dist.shape[0])
        min_dist, min_idx = knn.values, knn.indices
        count = 1
        while True:
            if len(self.visited) == dist.shape[0]:
                print("CANNOT FIND IN THE ENTIRE DATASET !!!")
                exit()
            if min_idx[count].item() not in self.visited:
                self.visited[min_idx[count].item()] = True
                break
            count += 1

        return self.features[min_idx[count].item()], self.perf_labels[min_idx[count].item()], self.origin_embedding[min_idx[count].item()]


class Policy(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, input):
        x = F.relu(self.fc1(input))
        out = self.fc2(x)
        return out


class Policy_LSTM(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(Policy_LSTM, self).__init__()
        self.lstm = torch.nn.LSTMCell(input_size=hidden_dim1, hidden_size=hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, hidden_dim1)
        self.saved_log_probs = []
        self.rewards = []
        self.hx = None
        self.cx = None

    def forward(self, input):
        if self.hx is None and self.cx is None:
            self.hx, self.cx = self.lstm(input)
        else:
            self.hx, self.cx = self.lstm(input, (self.hx, self.cx))
        out = self.fc(self.hx)
        return out


class Arch2vec_RL_NAS(NASBase):

    def __init__(
        self,
        search_space: SearchSpaceBase, 
        trainer: TrainerBase, 
        config: dict, 
        **kwargs
    ):
        super().__init__(search_space=search_space, trainer=trainer, config=config, **kwargs)
        self.config = config
        if self.config.isarch2vecrl == True:
            self.dataset_name = self.config.dataset_name
            self.num_epochs = self.config.epochs
            ### model config
            self.trajectory_bs = self.config.arch2vecrl.trajectory_bs
            self.embed_dim = self.config.arch2vecrl.embed_dim
            self.input_dim = self.config.arch2vecrl.input_dim
            self.hidden_dim = self.config.arch2vecrl.hidden_dim
            self.num_hops = self.config.arch2vecrl.num_hops
            self.num_mlp_layers = self.config.arch2vecrl.num_mlp_layers
            self.dropout = self.config.arch2vecrl.dropout
            self.arch2vec_cfg = self.config.arch2vecrl.arch2vec_cfg
            self.baseline_decay = self.config.arch2vecrl.baseline_decay
            self.env = Env(self.dataset_name, self.embed_dim, self.input_dim, self.hidden_dim, self.num_hops, self.num_mlp_layers, self.dropout, configs[self.arch2vec_cfg])
        else:
            print("The config \'isarch2vecrl\' is not True, please check!")
            exit()
    
    def select_action(self, state, policy):
        """
        MVN based action selection.
        :param state: 1 x dim
        :param policy: policy network
        :return: selected action: 1 x dim
        """
        out = policy(state.view(1, state.shape[0]))
        mvn = MultivariateNormal(out, 1.0*torch.eye(state.shape[0]).cuda())
        action = mvn.sample()
        policy.saved_log_probs.append(torch.mean(mvn.log_prob(action)))
        return action


    def finish_episode(self, policy, optimizer):
        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards:
            R = r + self.baseline_decay * R
            returns.append(R)
        returns = torch.Tensor(policy.rewards)
        val, indices = torch.sort(returns)
        print("sorted validation reward:", val)
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.mean(torch.stack(policy_loss, dim=0))
        print("average reward: {}, policy loss: {}".format(sum(policy.rewards)/len(policy.rewards), policy_loss.item()))
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:] # to avoid active learning with increasing pool size
        del policy.saved_log_probs[:]
        policy.hx = None
        policy.cx = None
    
    def is_valid_gnn(self, embed):
        op, ad = self.env.model.decoder(embed.unsqueeze(0).cuda())
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

    def search(self):
        policy = Policy_LSTM(self.embed_dim, 128).cuda()
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        counter = 0
        epoch = 0
        state, perf, origin_embed = self.env.get_init_state()
        CURR_BEST_PERF = 0.
        CURR_BEST_OP = None
        CURR_BEST_LINK = None
        perf_trace = []
        
        while epoch < self.num_epochs or CURR_BEST_OP is None:
            for c in range(self.trajectory_bs):
                state = state.cuda()
                action = self.select_action(state, policy)
                state, reward, origin_embed = self.env.step(action)
                is_valid, op_decode, ad_decode, link_decode = self.is_valid_gnn(origin_embed)

                #### delete
                if "skip" in op_decode or "fc" in op_decode:
                    continue
                ####

                policy.rewards.append(reward)
                counter += 1
                print('counter: {}, reward: {}'.format(counter, reward))
                if reward > CURR_BEST_PERF and is_valid:
                    CURR_BEST_PERF = reward
                    CURR_BEST_OP = op_decode
                    CURR_BEST_LINK = link_decode
                    
                perf_trace.append(float(CURR_BEST_PERF))
            epoch += 1
            print("epoch: ", epoch)
            print("curr_best_perf: ", CURR_BEST_PERF)
            print("op: ", CURR_BEST_OP[1:-1])
            print("adj: ", CURR_BEST_LINK)

            self.finish_episode(policy, optimizer)
        print(CURR_BEST_PERF)
        # reconstruct GNN        
        return self.search_space.to_gnn(CURR_BEST_OP[1:-1])
    
    def fit(self, data) -> GNNBase:
        self.best_model = self.search()
        return self.best_model


model_factory["arch2vec_rl_nas"] = Arch2vec_RL_NAS