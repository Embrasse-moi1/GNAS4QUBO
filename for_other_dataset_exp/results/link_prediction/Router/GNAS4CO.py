import torch
from easydict import EasyDict as edict
from for_other_dataset_exp.llm4gnas.utils.get_lp_data import *
criterion = torch.nn.functional.cross_entropy
from sklearn.metrics import roc_auc_score
from for_other_dataset_exp.llm4gnas.search_space import *
from for_other_dataset_exp.llm4gnas.args import *
def train(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    model.to(device)
    model.train()
    # setting of data shuffling move to dataloader creation
    for step in tqdm(range(1, 201)):
        for batch in dataloader:
            batch = batch.to(device)
            label = batch.y
            prediction = model(batch)
            loss = criterion(prediction, label, reduction='mean')
            # loss.backward()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute loss:
        loss = criterion(predictions, labels, reduction='mean').item()
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item()/labels.shape[0]
        # compute auc:
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        multi_class = 'ovr'
        if predictions.size(1) == 2:
            predictions = predictions[:, 1]
            multi_class = 'raise'
        auc = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy(), multi_class=multi_class)
    return loss, acc, auc

def eval_model(model, dataloader, device, return_predictions=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels.append(batch.y)
            prediction = model(batch)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
    if not return_predictions:
        loss, acc, auc = compute_metric(predictions, labels)
        return loss, acc, auc
    else:
        return predictions


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="description")
    register_args(parser)
    register_args_autogel(parser, lp="LinkPredict")
    args = parser.parse_args()
    args.task_name = "LinkPredict"
    args.in_dim = 53
    args.out_dim = 2
    args.dataset_name = "router"

    # 加载数据
    (G, labels), task = read_file(args, args.dataset_name)
    dataloaders, out_features = get_data(G, task=task, labels=labels, args=args)
    train_loader, val_loader, test_loader = dataloaders
    desc = "{layer1:{ agg:max, combine:sum, act:relu, layer_connect:stack}; layer2:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_sum}; layer_agg:concat;}"
    search_space = Autogel_Space(args)
    model = search_space.to_gnn(desc=desc)
    model.to(device)
    train(test_loader, model)
    test_loss, test_auc, test_acc = eval_model(model, test_loader, device)
    print("test_acc: ", test_acc, "test_auc: ", test_auc)