import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        real_part = self.linear_real(x.real) - self.linear_imag(x.imag)
        imag_part = self.linear_imag(x.real) + self.linear_real(x.imag)
        out = torch.complex(real_part, imag_part)
        return out

def complex_softmax(inp, dim=0):
    magnitudes = torch.abs(inp)

    max_magnitudes = torch.max(magnitudes, dim=dim, keepdim=True).values
    exp_magnitudes = torch.exp(magnitudes - max_magnitudes)
    sum_exp_magnitudes = torch.sum(exp_magnitudes, dim=dim, keepdim=True)

    softmax_magnitudes = exp_magnitudes / sum_exp_magnitudes

    phase_factors = inp / (magnitudes + 1e-8)

    real_part = softmax_magnitudes * phase_factors.real
    imag_part = softmax_magnitudes * phase_factors.imag

    return torch.complex(real_part, imag_part)

def complex_relu(inp):
    real_part = torch.relu(inp.real)
    imag_part = torch.relu(inp.imag)

    return torch.complex(real_part, imag_part)

class ComplexLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(ComplexLayerNorm, self).__init__()
        self.layer_norm_real = nn.LayerNorm(normalized_shape)
        self.layer_norm_imag = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        real_part = self.layer_norm_real(x.real)
        imag_part = self.layer_norm_imag(x.imag)
        x_normalized = torch.complex(real_part, imag_part)
        return x_normalized
