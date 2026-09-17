import torch
import numpy as np
from torch import nn
#import modules.observers as observers
import modules.functions as functions
import modules.quantization as quantization

from mqbench.observer import MSEObserver,EMAMSEObserver,MinMaxObserver

# Conv Type list:
# - 1 : standard convolution
# - 2 : quantized convolution no error
# - 3 : quantized convolution error STE gradient
# - 4 : quantized convolution error aware gradient

#TODO Rendere parametrico anche matrice di approx_mult
class Conv2d_custom(nn.Conv2d):
    def __init__(self,channel_in,
                channel_out,
                kernel_size,
                stride,
                padding,
                bias,
                conv_type,
                bit_width,
                multiplier_matrix,
                signed = False,
                name = None):
        
        super().__init__(channel_in,channel_out,kernel_size,stride,padding,bias = bias)
        
        self.register_buffer('activation_scale', torch.tensor(1.0))
        self.register_buffer('activation_zp_neg', torch.tensor(0.0))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zp_neg', torch.tensor(0.0))
        self.register_buffer('out_scale', torch.tensor(1.0))
        self.register_buffer('out_zp_neg', torch.tensor(0.0))
        
        self.signed = signed
        if not 1 <= bit_width <= 8:
            raise ValueError("bit_width must be between 1 and 8")

        quant_dtype = torch.qint8 if signed else torch.quint8
        quant_min = -(2 ** (bit_width - 1)) if signed else 0
        quant_max = 2 ** (bit_width - 1) - 1 if signed else 2 ** bit_width - 1
        observer_args = {
            "dtype": quant_dtype,
            "qscheme": torch.per_tensor_affine,
            "quant_min": quant_min,
            "quant_max": quant_max,
        }
        self.activation_observer = EMAMSEObserver(**observer_args)
        self.weight_observer = MSEObserver(**observer_args)
        self.output_observer = EMAMSEObserver(**observer_args)
        self.bit_width = bit_width
        self.multiplier_matrix = multiplier_matrix
        self.calibrating = False
        
        self.name = name
        self.conv_type = conv_type
        if(conv_type == 1):
            self.conv2d_op = None
        elif(conv_type == 2):
            self.conv2d_op = functions.QuantizedConv2d
        elif(conv_type == 3):
            self.conv2d_op = functions.ApproxConv2dSTE
        elif(conv_type == 4):
            self.conv2d_op = functions.ApproxConv2d
        elif(conv_type == 5):
            self.conv2d_op = functions.StatsQuantizedConv2d
        else:
            raise(NotImplementedError) 

    def freeze_qparams(self):
        act_scale, act_zp = self.activation_observer.calculate_qparams()
        w_scale, w_zp = self.weight_observer.calculate_qparams()
        out_scale, out_zp = self.output_observer.calculate_qparams()

        self.activation_scale.copy_(act_scale.squeeze())
        self.activation_zp_neg.copy_(-act_zp.squeeze())
        self.weight_scale.copy_(w_scale.squeeze())
        self.weight_zp_neg.copy_(-w_zp.squeeze())
        self.out_scale.copy_(out_scale.squeeze())
        self.out_zp_neg.copy_(-out_zp.squeeze())


        self.calibrating = False

    """ if(self.training and self.conv_type == 5):
            self.conv2d_op = functions.QuantizedConv2d
        elif(not self.training and (self.conv_type == 5)):
            self.conv2d_op = functions.StatsQuantizedConv2d
        if(self.conv2d_op == None):
            return nn.functional.conv2d(input=input, 
                                        weight=self.weight,
                                        bias=self.bias,
                                        stride=self.stride,
                                        padding=self.padding)"""
    def forward(self, input):
        if self.conv_type == 1 or self.conv2d_op is None:
            return nn.functional.conv2d(input, self.weight, self.bias,
                                        self.stride, self.padding)

        if self.calibrating:
            self.activation_observer(input)
            self.weight_observer(self.weight)
            out = nn.functional.conv2d(input, self.weight, self.bias,
                                        self.stride, self.padding)
            self.output_observer(out)  
            return out                           
        if self.signed:
            print("NOT IMPLEMENTED YET")
            return
            """input_int = quantization.signed_quantization(input, self.activation_scale, self.activation_quant_max)
            weight_int = quantization.signed_quantization(self.weight, self.weight_scale, self.weight_quant_max)"""
        else:
            input_int = quantization.unsigned_quantization(input, self.activation_scale, self.activation_zp_neg, self.bit_width)
            weight_int = quantization.unsigned_quantization(self.weight, self.weight_scale, self.weight_zp_neg, self.bit_width)   
        out =  self.conv2d_op.apply(input,
                            self.weight,
                            input_int,
                            weight_int,
                            self.bias, 
                            self.stride, 
                            self.padding,
                            self.activation_scale,
                            self.weight_scale,
                            self.activation_zp_neg,
                            self.weight_zp_neg,
                            self.signed,
                            self.bit_width,
                            self.name,
                            self.multiplier_matrix)
        return quantization.simulate_industrial_rescaling(out, self.out_scale, self.out_zp_neg, self.bit_width)