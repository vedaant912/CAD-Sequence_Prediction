import os
import torch


def set_up_causal_mask(seq_len, device):
    """Defines the triangular mask used in transformers.

    This mask prevents decoder from attending the tokens after the current one.

    Arguments:
        seq_len (int): Maximum length of input sequence
        device: Device on which to map the created tensor mask
    Returns:
        mask (torch.Tensor): Created triangular mask
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    mask.requires_grad = False
    return mask