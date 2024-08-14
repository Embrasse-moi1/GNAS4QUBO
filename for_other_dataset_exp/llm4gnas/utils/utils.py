import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()


def compute_metric(predictions, labels):
    with torch.no_grad():
        # compute acc:
        correct_predictions = (torch.argmax(predictions, dim=1) == labels)
        acc = correct_predictions.sum().cpu().item() / labels.shape[0]
    return acc
