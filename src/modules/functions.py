import torch
import numpy as np
import os
from torch import nn

heat_map_path = "./heat_maps/npy_matrix/"
#TODO Adapt backpropagation custom methods to make it work with signed and unsigned
#La parte python dovrebbe essere corretta
# ********************* Backpropagation Custom Methods *********************

def gradient_error_inputs(input, kernel, grad_output, stride, padding, weight_zp, bit_width, signed):
    derivative_matrix_x = torch.from_numpy(np.load('der_x.npy')).float().to("cuda")
    
    _, _, start_height, start_width = input.size()
    out_channels, _, kernel_height, kernel_width = kernel.size()

    input_unfolded = nn.functional.unfold(input, kernel_size=(kernel_height, kernel_width), stride=stride).transpose(1, 2)
    kernel_flatten = kernel.view(out_channels, -1).T

    output = torch.ops.mat_mul.derivate_input(
        input_unfolded.contiguous(), kernel_flatten.contiguous(), derivative_matrix_x.contiguous(), grad_output.contiguous(), weight_zp, bit_width, signed
    )

    output = output.transpose(1, 2)
    output = nn.functional.fold(
        output,
        output_size=(start_height - (2 * padding[0]), start_width - (2 * padding[1])),
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding
    )

    del derivative_matrix_x, input_unfolded, kernel_flatten
    return output


def gradient_error_weights(input, kernel, grad_output, stride, activation_zp, bit_width, signed):
    derivative_matrix_y = torch.from_numpy(np.load('der_y.npy')).float().to("cuda")
    _, in_channels, _, _ = input.size()
    out_channels, _, kernel_height, kernel_width = kernel.size()

    input_unfolded = nn.functional.unfold(input, kernel_size=(kernel_height, kernel_width), stride=stride).transpose(1, 2)
    kernel_flatten = kernel.view(out_channels, -1).T

    output = torch.ops.mat_mul.derivate_weight(
        input_unfolded.contiguous(), kernel_flatten.contiguous(), derivative_matrix_y.contiguous(), grad_output.contiguous(), activation_zp, bit_width, signed   
    )
    output = output.sum(dim=1).view(out_channels, in_channels, kernel_height, kernel_width)

    del derivative_matrix_y, input_unfolded, kernel_flatten
    return output


# ********************* Forward Methods  *********************

def approx_convolution(input, weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, multiplier_matrix):

    if multiplier_matrix is None:
        return quantized_convolution(
            input,
            weight,
            bias,
            stride,
            act_scale,
            weight_scale,
            activation_zp,
            weight_zp,
            signed,
            stats=False,
            bit_width=bit_width,
            name=None,
        )

    res_matrix = torch.from_numpy(np.load(multiplier_matrix)).float().to("cuda")
    batch_size, _, in_height, in_width = input.size()
    out_channels, _, weight_height, weight_width = weight.size()
    input_unfolded = nn.functional.unfold(input, kernel_size=(weight_height, weight_width), stride=stride)
    kernel_flatten = weight.view(out_channels, -1)
    output =  torch.ops.mat_mul.matmul_cuda(
        input_unfolded.transpose(1, 2).contiguous(), kernel_flatten.T.contiguous(), res_matrix.contiguous(), act_scale, activation_zp, weight_scale, weight_zp, bit_width, signed
    ).transpose(1, 2)
    output_height = (in_height - weight_height) // stride[0] + 1
    output_width = (in_width - weight_width) // stride[1] + 1
    output = output.view(batch_size, out_channels, output_height, output_width)

    if bias is not None:
        output.add_(bias.view(1, out_channels, 1, 1))
    del res_matrix
    return output

def quantized_convolution(input, weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed, stats = False, bit_width = 0, name = None):
    batch_size, _, in_height, in_width = input.size()
    out_channels, _, weight_height, weight_width = weight.size()

    heat_map = None
    if(stats):
        if os.path.isfile(heat_map_path + name + ".npy"):
            heat_map = torch.from_numpy(np.load(heat_map_path + name + ".npy")).float().to("cuda")
        else:
            heat_map = torch.zeros((out_channels, 2**bit_width, 2**bit_width), dtype = torch.float32).to("cuda")


    input_unfolded = nn.functional.unfold(input, kernel_size=(weight_height, weight_width), stride=stride)
    kernel_flatten = weight.view(out_channels, -1)
    #print(heat_map.dtype)
    output =  torch.ops.mat_mul.matmul_no_error_cuda(
        input_unfolded.transpose(1, 2).contiguous(), kernel_flatten.T.contiguous(), heat_map, act_scale, activation_zp, weight_scale, weight_zp, bit_width, signed
    ).transpose(1, 2)

    output_height = (in_height - weight_height) // stride[0] + 1
    output_width = (in_width - weight_width) // stride[1] + 1
    output = output.view(batch_size, out_channels, output_height, output_width)

    if(stats):
        heat_map = heat_map.to("cpu")
        np.save(heat_map_path + name + ".npy", heat_map)
    if bias is not None:
        output.add_(bias.view(1, out_channels, 1, 1))
    return output


#********************* Functions Definition *********************

class ApproxConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, _, multiplier_matrix):
        ctx.stride = stride
        ctx.padding = padding
        input_padded = nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))(int_input)
        ctx.save_for_backward(input_padded, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        ctx.bit_width = bit_width
        ctx.signed = signed
        if(not signed):
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
        return approx_convolution(input_padded, int_weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, multiplier_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except:
            pass
        bit_width = ctx.bit_width
        signed = ctx.signed
        stride, padding = ctx.stride, ctx.padding

        error_derivate_weights = gradient_error_weights(input, weight, grad_output, stride, activation_zp, bit_width, signed)
        grad_weight = (error_derivate_weights) * act_scale

        error_derivate_inputs = gradient_error_inputs(input, weight, grad_output, stride, padding, weight_zp, bit_width, signed)
        grad_input = (error_derivate_inputs) * weight_scale

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    
class ApproxConv2dSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, _, multiplier_matrix):
        ctx.stride = stride
        ctx.padding = padding
        input_padded = nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))(int_input)
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if(not signed):
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
        return approx_convolution(input_padded, int_weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, multiplier_matrix)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except:
            pass

        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding = ctx.stride, ctx.padding

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape , weight + weight_zp, grad_output, stride, padding
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    
class QuantizedConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, activation_zp, weight_zp, signed, _, __, ___):
        ctx.stride = stride
        ctx.padding = padding
        input_padded = nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))(int_input)
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if(not signed):
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
        return quantized_convolution(input_padded, int_weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except:
            pass
        
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding = ctx.stride, ctx.padding

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape , weight + weight_zp, grad_output, stride, padding
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None


class StatsQuantizedConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, activation_zp, weight_zp, signed, bit_width, name, _):
        ctx.stride = stride
        ctx.padding = padding
        input_padded = nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))(int_input)
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if(not signed):
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
        return quantized_convolution(input_padded, int_weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, signed, stats=True, bit_width= bit_width, name = name)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except:
            pass
        
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding = ctx.stride, ctx.padding

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape , weight + weight_zp, grad_output, stride, padding
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None
