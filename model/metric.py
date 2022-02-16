import torch
import torch.nn.functional as F
import numpy as np


def perplexity(output, target):
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    pp = F.nll_loss(output, target, ignore_index=0, reduction="sum")
    return pp / batch_size

def accuracy(output, target):
    correct = 0
    for output_str, target_str in zip(output, target):
        if output_str == target_str:
            correct += 1
    return correct / len(target)