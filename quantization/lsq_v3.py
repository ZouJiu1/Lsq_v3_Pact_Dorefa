import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
 
class FunLSQ(Function):
    @staticmethod
    def forward(ctx, values, scale, q_range, g, Qn, Qp):
        ctx.save_for_backward(values, scale)
        ctx.other = q_range, g, Qn, Qp
        values = Round.apply((values * q_range / scale).clamp(Qn, Qp)) / q_range * scale
        return values

    @staticmethod
    def backward(ctx, grad_weight):
        values, scale = ctx.saved_tensors
        q_range, g, Qn, Qp = ctx.other
        q_kk = values * q_range / scale
        q_w = values / scale

        smaller = (q_kk < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_kk > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        grad_scale = ((smaller * Qn / q_range + bigger * Qp / q_range + 
                between * Round.apply(q_kk) / q_range - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0) #?
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        return grad_weight, grad_scale, None, None, None, None

# A(特征)量化
class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.q_range = self.Qp - self.Qn
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.s.data = torch.tensor(1)
        # self.register_parameter('Ascale', self.s)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        #V3
        if self.init_state==0:
            self.g = 1.0 #1.0/math.sqrt(activation.numel() * self.Qp)
            # self.s.data = torch.mean(torch.abs(activation.detach()))*2/(math.sqrt(self.Qp))
            self.init_state += 1
        # elif self.init_state<self.batch_init:
        #     self.s.data = 0.9*self.s.data + 0.1*torch.mean(torch.abs(activation.detach()))*2/(math.sqrt(self.Qp))
        #     self.init_state += 1
        # elif self.init_state==self.batch_init:
        #     self.s.data = self.s.data # / self.q_range
        #     self.init_state += 1
        if self.a_bits == 32:
            output = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            # print("a: ", self.s, self.g)
            q_a = FunLSQ.apply(activation, self.s, self.q_range, self.g, self.Qn, self.Qp)
        return q_a

# W(权重)量化
class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, batch_init = 20):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.q_range = self.Qp - self.Qn
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.s.data = torch.tensor(0.01)
        # self.register_parameter('Wscale', self.s)
        self.init_state = 0

    # 量化/反量化
    def forward(self, weight):
        if self.init_state==0:
            self.g = 1.0 #1.0/math.sqrt(weight.numel() * self.Qp)
            # self.s.data = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
            self.init_state += 1
        # elif self.init_state<self.batch_init:
        #     self.s.data = 0.9*self.s.data + 0.1*torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
        #     self.init_state += 1
        # elif self.init_state==self.batch_init:
        #     self.s.data = self.s.data / self.q_range
        #     self.init_state += 1
        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            # print("w: ", self.s, self.g)
            w_q = FunLSQ.apply(weight, self.s, self.q_range, self.g, self.Qn, self.Qp)
        return w_q

class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False, 
                 batch_init = 20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        # print('input:',input.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False,
                 batch_init = 20):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False,
                 batch_init = 20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
                 quant_inference=False, all_positive=False,
                 batch_init = 20):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups,
                                                                bias=True,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups, bias=False,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, batch_init = batch_init)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, all_positive=all_positive, batch_init = batch_init)


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, batch_init = 20):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, all_positive=all_positive, 
                 batch_init = batch_init)
    return model