import torch
from torch.nn import functional as F
import numpy as np

eps = 1e-20

def get_lam(loss):
    bs = loss.size()[0]
    loss = (loss - loss.min()) / (loss.max() - loss.min())
    mask = np.random.normal(loc=1.0, scale=0.1, size=bs)

    coeff = int(bs * 0.3)
    top_value, _ = torch.topk(loss, k=coeff)
    for i in range(bs):
        if loss[i] >= top_value[-1]:
            mask[i] = max(mask[i], 2 - mask[i])
        else:
            mask[i] = min(mask[i], 2 - mask[i])

    return mask.reshape(bs, 1)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def log_loss(logits_p, prob_p, logits_q, prob_q):
    prob_p = prob_p  if prob_p is not None   else F.softmax(logits_p, dim=1)
    logq = F.log_softmax(logits_q, dim=1)   if logits_q is not None  else torch.log(prob_q + eps)
    return -torch.mean(torch.sum(prob_p.detach() * logq, dim=1))


def euclidean_metric(a, b):
    # distance
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n,m,-1)
    b = b.unsqueeze(0).expand(n,m,-1)
    logits = -((a-b)**2).sum(dim=2)
    return logits


def count_acc(logts, label):
    pred = torch.argmax(logts, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    # lambda_u = 25
    return 25*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch, warm_up)



