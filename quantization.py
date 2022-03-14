import torch
from torch import nn

def quantize(x, bits=8):
    """
    Quantize a tensor to the specified number of bits.
    """
    n = 2. ** bits - 1.
    a = torch.min(x)
    b = torch.max(x)
    s = (b - a) / n
    if a < 0 and b > 0:
        # make sure 0.0 is exactly representable
        da = a - s * torch.floor((a / s).float())
        a = a - da
        b = b - da
    q = torch.round(((torch.clamp(x, a, b) - a) / s)) * s + a
    return q, a, b, s

def quantize_conv_layer(conv, bits=8):
    """
    Quantize a convolutional layer.
    """
    with torch.no_grad():
        w, a, b, s = quantize(conv.weight, bits)
        conv.weight.data = w
        # conv.bias = (conv.bias - a) / s
    return a, b, s

def quantize_tensor(t, bits=8):
    """
    Quantize an arbitrary tensor.
    """
    with torch.no_grad():
        w, a, b, s = quantize(t, bits)
        t.data = w
    return a, b, s


