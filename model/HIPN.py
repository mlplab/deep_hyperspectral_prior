# coding: utf-8
'''
Hyperspectral Image Prior Network Model
'''


import torch
from torchsummary import summary
from .layers import swish, mish, HSI_prior_block


class HSI_Network(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, activation='relu', output_norm=None, **kwargs):
        super(HSI_Network, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = [HSI_prior_block(output_ch, output_ch, feature=feature) for _ in range(block_num)]
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        # self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1)
        residual_block = [torch.nn.Conv2d(output_ch, output_ch, 3, 1, 1) for _ in range(block_num)]
        self.residual_block = torch.nn.Sequential(*residual_block)
        # self.shortcut = torch.nn.Sequential(*shortcut)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = x
        for hsi, res in zip(self.hsi_block, self.residual_block):
            x_hsi = hsi(x)
            x_res = res(x)
            # x_shortcut = shortcut(h)
            x = x_res + x_hsi + x_in
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x


class HSI_Network_share(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, activation='relu', output_norm=None, **kwargs):
        super(HSI_Network_share, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = [HSI_prior_block(output_ch, output_ch, feature=feature) for _ in range(block_num)]
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = x
        for hsi_block in self.hsi_block:
            x_hsi = hsi_block(x)
            x_res = self.residual_block(x)
            x = x_res + x_hsi + x_in
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x