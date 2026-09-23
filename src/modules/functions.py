import os
import numpy as np
import torch
import torch.nn as nn

heat_map_path = "./heat_maps/npy_matrix/"

# ********************* Backpropagation Custom Methods *********************

def gradient_error_inputs(input, kernel, grad_output, stride, padding, weight_zp, bit_width, signed, dilation=1, groups=1):
    derivative_matrix_x = torch.from_numpy(np.load('der_x.npy')).float().to("cuda")
    
    batch_size, in_channels, start_height, start_width = input.size()
    out_channels, in_channels_per_group, kernel_height, kernel_width = kernel.size()
    
    out_channels_per_group = out_channels // groups
    
    # Split input, kernel, e grad_output in gruppi
    input_groups = torch.chunk(input, groups, dim=1)
    kernel_groups = torch.chunk(kernel, groups, dim=0)
    grad_output_groups = torch.chunk(grad_output, groups, dim=1)
    
    output_groups = []
    
    for g in range(groups):
        input_g = input_groups[g]
        kernel_g = kernel_groups[g]
        grad_output_g = grad_output_groups[g]

        input_unfolded = nn.functional.unfold(
            input_g, 
            kernel_size=(kernel_height, kernel_width), 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        ).transpose(1, 2)
        
        kernel_flatten = kernel_g.view(out_channels_per_group, -1).T

        out_g = torch.ops.mat_mul.derivate_input(
            input_unfolded.contiguous(), 
            kernel_flatten.contiguous(), 
            derivative_matrix_x.contiguous(), 
            grad_output_g.contiguous(), 
            weight_zp, 
            bit_width, 
            signed
        ).transpose(1, 2)

        out_g = nn.functional.fold(
            out_g,
            output_size=(start_height, start_width),
            kernel_size=(kernel_height, kernel_width),
            dilation=dilation,
            padding=padding,
            stride=stride
        )
        output_groups.append(out_g)

    output = torch.cat(output_groups, dim=1)
    del derivative_matrix_x
    return output


def gradient_error_weights(input, kernel, grad_output, stride, activation_zp, bit_width, signed, padding=0, dilation=1, groups=1):
    derivative_matrix_y = torch.from_numpy(np.load('der_y.npy')).float().to("cuda")
    
    batch_size, in_channels, _, _ = input.size()
    out_channels, in_channels_per_group, kernel_height, kernel_width = kernel.size()
    out_channels_per_group = out_channels // groups

    input_groups = torch.chunk(input, groups, dim=1)
    kernel_groups = torch.chunk(kernel, groups, dim=0)
    grad_output_groups = torch.chunk(grad_output, groups, dim=1)

    weight_grad_groups = []

    for g in range(groups):
        input_g = input_groups[g]
        kernel_g = kernel_groups[g]
        grad_output_g = grad_output_groups[g]

        input_unfolded = nn.functional.unfold(
            input_g, 
            kernel_size=(kernel_height, kernel_width), 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        ).transpose(1, 2)
        
        kernel_flatten = kernel_g.view(out_channels_per_group, -1).T

        out_g = torch.ops.mat_mul.derivate_weight(
            input_unfolded.contiguous(), 
            kernel_flatten.contiguous(), 
            derivative_matrix_y.contiguous(), 
            grad_output_g.contiguous(), 
            activation_zp, 
            bit_width, 
            signed   
        )
        out_g = out_g.sum(dim=1).view(out_channels_per_group, in_channels_per_group, kernel_height, kernel_width)
        weight_grad_groups.append(out_g)

    output = torch.cat(weight_grad_groups, dim=0)
    del derivative_matrix_y
    return output


# ********************* Forward Methods  *********************

def approx_convolution(input, weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp, 
                       signed, bit_width, multiplier_matrix, dilation=1, groups=1):

    if multiplier_matrix is None:
        return quantized_convolution(
            input, weight, bias, stride, act_scale, weight_scale,
            activation_zp, weight_zp, signed, stats=False,
            bit_width=bit_width, name=None, dilation=dilation, groups=groups
        )

    res_matrix = torch.from_numpy(np.load(multiplier_matrix)).float().to("cuda")
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels, in_channels_per_group, weight_height, weight_width = weight.size()
    
    out_channels_per_group = out_channels // groups

    # Calcolo delle dimensioni di output considerando la dilation
    eff_kernel_h = weight_height + (weight_height - 1) * (dilation[0] - 1) if isinstance(dilation, tuple) else weight_height + (weight_height - 1) * (dilation - 1)
    eff_kernel_w = weight_width + (weight_width - 1) * (dilation[1] - 1) if isinstance(dilation, tuple) else weight_width + (weight_width - 1) * (dilation - 1)
    
    str_h, str_w = (stride, stride) if isinstance(stride, int) else stride

    output_height = (in_height - eff_kernel_h) // str_h + 1
    output_width = (in_width - eff_kernel_w) // str_w + 1

    input_groups = torch.chunk(input, groups, dim=1)
    weight_groups = torch.chunk(weight, groups, dim=0)
    output_groups = []

    for g in range(groups):
        input_unfolded = nn.functional.unfold(
            input_groups[g], 
            kernel_size=(weight_height, weight_width), 
            dilation=dilation, 
            stride=stride
        )
        kernel_flatten = weight_groups[g].view(out_channels_per_group, -1)

        out_g = torch.ops.mat_mul.matmul_cuda(
            input_unfolded.transpose(1, 2).contiguous(), 
            kernel_flatten.T.contiguous(), 
            res_matrix.contiguous(), 
            act_scale, activation_zp, weight_scale, weight_zp, bit_width, signed
        ).transpose(1, 2)

        out_g = out_g.view(batch_size, out_channels_per_group, output_height, output_width)
        output_groups.append(out_g)

    output = torch.cat(output_groups, dim=1)

    if bias is not None:
        output.add_(bias.view(1, out_channels, 1, 1))
    del res_matrix
    return output


def quantized_convolution(input, weight, bias, stride, act_scale, weight_scale, activation_zp, weight_zp,
                           signed, stats=False, bit_width=0, name=None, shift_bits=0, dilation=1, groups=1):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels, in_channels_per_group, weight_height, weight_width = weight.size()
    out_channels_per_group = out_channels // groups

    heat_map = None
    if stats:
        if os.path.isfile(heat_map_path + name + ".npy"):
            heat_map = torch.from_numpy(np.load(heat_map_path + name + ".npy")).float().to("cuda")
        else:
            heat_map = torch.zeros((out_channels, 2**bit_width, 2**bit_width), dtype=torch.float32).to("cuda")

    if shift_bits > 0:
        input_dtype = input.dtype
        weight_dtype = weight.dtype
        int_dtype = torch.int32

        input_int = input.to(int_dtype)
        weight_int = weight.to(int_dtype)
        input_int = torch.div(input_int, 2**shift_bits, rounding_mode='trunc') if signed \
            else torch.bitwise_right_shift(input_int, shift_bits)
        weight_int = torch.div(weight_int, 2**shift_bits, rounding_mode='trunc') if signed \
            else torch.bitwise_right_shift(weight_int, shift_bits)

        input = input_int.to(input_dtype)
        weight = weight_int.to(weight_dtype)

    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    str_h, str_w = (stride, stride) if isinstance(stride, int) else stride

    eff_kernel_h = weight_height + (weight_height - 1) * (dil_h - 1)
    eff_kernel_w = weight_width + (weight_width - 1) * (dil_w - 1)

    output_height = (in_height - eff_kernel_h) // str_h + 1
    output_width = (in_width - eff_kernel_w) // str_w + 1

    input_groups = torch.chunk(input, groups, dim=1)
    weight_groups = torch.chunk(weight, groups, dim=0)
    output_groups = []

    for g in range(groups):
        input_unfolded = nn.functional.unfold(
            input_groups[g], 
            kernel_size=(weight_height, weight_width), 
            dilation=dilation, 
            stride=stride
        )
        kernel_flatten = weight_groups[g].view(out_channels_per_group, -1)

        heat_map_g = heat_map[g * out_channels_per_group : (g + 1) * out_channels_per_group] if stats else None

        out_g = torch.ops.mat_mul.matmul_no_error_cuda(
            input_unfolded.transpose(1, 2).contiguous(), 
            kernel_flatten.T.contiguous(),
            heat_map_g, act_scale, activation_zp, weight_scale, weight_zp, bit_width, signed, shift_bits
        ).transpose(1, 2)

        out_g = out_g.view(batch_size, out_channels_per_group, output_height, output_width)
        output_groups.append(out_g)

    output = torch.cat(output_groups, dim=1)

    if stats:
        heat_map = heat_map.to("cpu")
        np.save(heat_map_path + name + ".npy", heat_map)
    if bias is not None:
        output.add_(bias.view(1, out_channels, 1, 1))
    return output


# ********************* Functions Definition *********************

class ApproxConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, 
                activation_zp, weight_zp, signed, bit_width, name, multiplier_matrix, shift_bits=0, dilation=1, groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        input_padded = nn.ZeroPad2d((pad_w, pad_w, pad_h, pad_h))(int_input)
        
        ctx.save_for_backward(input_padded, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        ctx.bit_width = bit_width
        ctx.signed = signed
        if not signed:
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
            
        return approx_convolution(
            input_padded, int_weight, bias, stride, act_scale, weight_scale, 
            activation_zp, weight_zp, signed, bit_width, multiplier_matrix, 
            dilation=dilation, groups=groups
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except AttributeError:
            pass
            
        bit_width = ctx.bit_width
        signed = ctx.signed
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups

        error_derivate_weights = gradient_error_weights(
            input, weight, grad_output, stride, activation_zp, bit_width, signed, 
            padding=0, dilation=dilation, groups=groups
        )
        grad_weight = error_derivate_weights * act_scale

        error_derivate_inputs = gradient_error_inputs(
            input, weight, grad_output, stride, padding, weight_zp, bit_width, signed, 
            dilation=dilation, groups=groups
        )
        grad_input = error_derivate_inputs * weight_scale

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class ApproxConv2dSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, 
                activation_zp, weight_zp, signed, bit_width, name, multiplier_matrix, shift_bits=0, dilation=1, groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        input_padded = nn.ZeroPad2d((pad_w, pad_w, pad_h, pad_h))(int_input)
        
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if not signed:
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
            
        return approx_convolution(
            input_padded, int_weight, bias, stride, act_scale, weight_scale, 
            activation_zp, weight_zp, signed, bit_width, multiplier_matrix, 
            dilation=dilation, groups=groups
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except AttributeError:
            pass

        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding, dilation, groups
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape, weight + weight_zp, grad_output, stride, padding, dilation, groups
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class QuantizedConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, 
                activation_zp, weight_zp, signed, _, __, ___, shift_bits=0, dilation=1, groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        input_padded = nn.ZeroPad2d((pad_w, pad_w, pad_h, pad_h))(int_input)
        
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if not signed:
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
            
        return quantized_convolution(
            input_padded, int_weight, bias, stride, act_scale, weight_scale, 
            activation_zp, weight_zp, signed, shift_bits=shift_bits, 
            dilation=dilation, groups=groups
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except AttributeError:
            pass
        
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding, dilation, groups
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape, weight + weight_zp, grad_output, stride, padding, dilation, groups
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class StatsQuantizedConv2d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, int_input, int_weight, bias, stride, padding, act_scale, weight_scale, 
                activation_zp, weight_zp, signed, bit_width, name, _, shift_bits=0, dilation=1, groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
        input_padded = nn.ZeroPad2d((pad_w, pad_w, pad_h, pad_h))(int_input)
        
        ctx.save_for_backward(int_input, int_weight, bias)
        ctx.act_scale = act_scale
        ctx.weight_scale = weight_scale
        if not signed:
            ctx.activation_zp = activation_zp
            ctx.weight_zp = weight_zp
            
        return quantized_convolution(
            input_padded, int_weight, bias, stride, act_scale, weight_scale, 
            activation_zp, weight_zp, signed, stats=True, bit_width=bit_width, 
            name=name, shift_bits=shift_bits, dilation=dilation, groups=groups
        )
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        activation_zp, weight_zp = 0, 0
        try:
            activation_zp = ctx.activation_zp
            weight_zp = ctx.weight_zp
        except AttributeError:
            pass
        
        act_scale, weight_scale = ctx.act_scale, ctx.weight_scale
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups

        grad_weight = act_scale * torch.nn.grad.conv2d_weight(
            input + activation_zp, weight.shape, grad_output, stride, padding, dilation, groups
        )
        grad_input = weight_scale * torch.nn.grad.conv2d_input(
            input.shape, weight + weight_zp, grad_output, stride, padding, dilation, groups
        )   

        return grad_input, grad_weight, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None