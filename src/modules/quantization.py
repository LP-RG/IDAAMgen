import torch

def signed_quantization(x, s, qmax):
        qmax = qmax.to("cuda")
        x_affine = x / s
        qmin = -qmax
        x_int = torch.clamp(torch.round(x_affine),min = qmin, max = qmax)
        return x_int

def unsigned_quantization(x, s, zpn, bit_width):
        x_affine = x / s - zpn
        x_int = torch.clamp((torch.round(x_affine)), 0, 2**bit_width - 1)
        return x_int

import numpy as np


def simulate_industrial_rescaling(out_fp32, out_scale, out_zp, bit_width):
    q_out = out_fp32 / out_scale - out_zp

    q_clamped = torch.clamp(torch.round(q_out), 0, 2**bit_width - 1)

    rescaled_fp32 = (q_clamped + out_zp) * out_scale

    return rescaled_fp32