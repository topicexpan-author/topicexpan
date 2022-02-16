import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.1):
    """
    output: a (batch_size, num_classes) tensor
    target: a (batch_size, ) tensor of dtype long
    """
    output = torch.softmax(output/temperature, dim=0)
    output = torch.gather(output, dim=1, index=target[:, None])
    loss = - torch.log(output + 1e-12)
    return loss.sum()

def nll_loss(output, target):
    """
    output: a (batch_size, sequence_length, vocab_size) tensor
    target: a (batch_size, sequence_length) tensor
    """
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    return F.nll_loss(output, target, ignore_index=0, reduction="sum")
