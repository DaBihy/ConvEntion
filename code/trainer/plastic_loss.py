
import numpy as np
import torch
import torch.nn.functional as F


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)

def loss_fn(preds, target, num_class, class_weight, wtable):
    # class_weight = torch.from_numpy(class_weight).type(preds.type())
    wtable = torch.from_numpy(wtable).type(preds.type())
    y_ohe = torch.zeros(
        target.size(0), num_class, requires_grad=False
    ).type(preds.type()).scatter(1, target.reshape(-1, 1), 1)
    preds = F.softmax(preds, dim=1)
    preds = torch.clamp(preds, 1e-15, 1-1e-15)
    prod = torch.sum(torch.log(preds) * y_ohe, dim=0)
    prod = prod * class_weight / wtable / target.size(0)
    loss = -torch.sum(prod) / torch.sum(class_weight)
    return loss