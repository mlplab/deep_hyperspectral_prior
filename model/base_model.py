# coding: utf-8


import torch


class CNN_Block(torch.nn.Module):

    def __init__(self, in_feature, out_feature, kernel=3, pool=True, num_layer=3):
        super(CNN_Block, self).__init__()
        layers = []
        features = [in_feature] + [out_feature for _ in range(num_layer)]
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(
                features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        if pool is True:
            layers.append(torch.nn.MaxPool2d(2, 2))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CNN_Block_for_UNet(CNN_Block):

    def __init__(self, in_feature, out_feature, kernel=3, pool=True, num_layer=3):
        super(CNN_Block_for_UNet, self).__init__(
            in_feature, out_feature, kernel, pool, num_layer)
        layers = []
        features = [in_feature] + [out_feature for _ in range(num_layer)]
        if pool is True:
            layers.append(torch.nn.MaxPool2d(2, 2))
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(
                features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class D_CNN_Block(torch.nn.Module):

    def __init__(self, in_feature, out_feature, kernel=3, num_layer=3):
        super(D_CNN_Block, self).__init__()
        layers = [torch.nn.ConvTranspose2d(
            in_feature, out_feature, kernel_size=2, stride=2)]
        features = [out_feature for _ in range(num_layer)]
        for i in range(len(features) - 1):
            layers.append(torch.nn.Conv2d(
                features[i], features[i + 1], kernel_size=kernel, padding=1))
            layers.append(torch.nn.BatchNorm2d(features[i + 1]))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block, self).__init__()
        layer = []
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(input_ch))
        layer.append(torch.nn.ReLU())
        layer.append(torch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Conv_Block_UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride, padding, norm=True):
        super(Conv_Block_UNet, self).__init__()
        layer = []
        layer.append(torch.nn.Conv2d(input_ch, output_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding))
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        # layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return torch.nn.functional.relu(self.layer(x))


class D_Conv_Block(torch.nn.Module):

    def __init__(self, input_ch, output_ch, norm=True):
        super(D_Conv_Block, self).__init__()
        # layer = []
        layer = [torch.nn.ConvTranspose2d(
            input_ch, output_ch, kernel_size=2, stride=2)]
        # layer.append(torch.nn.Conv2d(input_ch, output_ch,
        #                              kernel_size=kernel_size, stride=stride,
        #                              padding=padding))
        if norm is True:
            layer.append(torch.nn.BatchNorm2d(output_ch))
        layer.append(torch.nn.ReLU())
        self.layer = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Bottoleneck(torch.nn.Module):

    def __init__(self, input_ch, k):
        super(Bottoleneck, self).__init__()
        bottoleneck = []
        bottoleneck.append(Conv_Block(input_ch, 128, 1, 1, 0))
        bottoleneck.append(Conv_Block(128, k, 3, 1, 1))
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        return self.bottoleneck(x)


class DenseBlock(torch.nn.Module):

    def __init__(self, input_ch, k, layer_num):
        super(DenseBlock, self).__init__()
        bottoleneck = []
        for i in range(layer_num):
            bottoleneck.append(Bottoleneck(input_ch, k))
            input_ch += k
        self.bottoleneck = torch.nn.Sequential(*bottoleneck)

    def forward(self, x):
        for i, bottoleneck in enumerate(self.bottoleneck):
            growth = bottoleneck(x)
            x = torch.cat((x, growth), dim=1)
        return x


class TransBlock(torch.nn.Module):

    def __init__(self, input_ch, compress=.5):
        super(TransBlock, self).__init__()
        self.conv1_1 = Conv_Block(input_ch, int(
            input_ch * compress), 1, 1, 0, norm=False)
        self.ave_pool = torch.nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.ave_pool(x)
        return x


# class SpectralNorm(nn.Module):
#     def __init__(self, module, name='weight', power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()
#
#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")
#
#         height = w.data.shape[0]
#         _w = w.view(height, -1)
#         for _ in range(self.power_iterations):
#             v = l2normalize(torch.matmul(_w.t(), u))
#             u = l2normalize(torch.matmul(_w, v))
#
#         sigma = u.dot((_w).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))
#
#     def _made_params(self):
#         try:
#             getattr(self.module, self.name + "_u")
#             getattr(self.module, self.name + "_v")
#             getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False
#
#     def _make_params(self):
#         w = getattr(self.module, self.name)
#
#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]
#
#         u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = Parameter(w.data)
#
#         del self.module._parameters[self.name]
#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)
#
#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)
#


# class SA_Block(torch.nn.Module):
#
#     def __init__(self, input_ch):
#         super(SA_Block, self).__init__()
#         self.theta = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
#         self.phi = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
#         self.g = torch.nn.Conv2d(input_ch, input_ch // 2, 1, 1, 0)
#         self.attn = torch.nn.Conv2d(input_ch // 2, input_ch, 1, 1, 0)
#         self.sigma_ratio = torch.nn.Parameter(
#             torch.zeros(1), requires_grad=True)
#
#     def forward(self, x):
#         batch_size, ch, h, w = x.size()
#         # theta path (first conv block)
#         theta = self.theta(x)
#         theta = theta.view(batch_size, ch // 8, h *
#                            w).permute((0, 2, 1))  # (bs, HW, CH // 8)
#         # phi path (second conv block)
#         phi = self.phi(x)
#         phi = torch.nn.functional.max_pool2d(phi, 2)
#         phi = phi.view(batch_size, ch // 8, h * w //
#                        4)  # (bs, CH // 8, HW // 4)
#         # attention path (theta and phi)
#         attn = torch.bmm(theta, phi)  # (bs, HW, HW // 4)
#         attn = torch.nn.functional.softmax(attn, dim=-1)
#         # g path (third conv block)
#         g = self.g(x)
#         g = torch.nn.functional.max_pool2d(g, 2)
#         # (bs, HW // 4, CH // 2)
#         g = g.view(batch_size, ch // 2, h * w // 4).permute((0, 2, 1))
#         # attention map (g and attention path)
#         attn_g = torch.bmm(attn, g)  # (bs, HW, CH // 2)
#         attn_g = attn_g.permute((0, 2, 1)).view(
#             batch_size, ch // 2, h, w)  # (bs, CH // 2, H, W)
#         attn_g = self.attn(attn_g)
#         return x + self.sigma_ratio * attn_g


class SA_Block(torch.nn.Module):

    def __init__(self, input_ch):
        super(SA_Block, self).__init__()
        self.theta = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.phi = torch.nn.Conv2d(input_ch, input_ch // 8, 1, 1, 0)
        self.g = torch.nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        # self.attn = torch.nn.Conv2d(input_ch // 2, input_ch, 1, 1, 0)
        self.sigma_ratio = torch.nn.Parameter(
            torch.zeros(1), requires_grad=True)

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        # theta path (first conv block)
        theta = self.theta(x)
        theta = theta.view(batch_size, ch // 8, h *
                           w).permute((0, 2, 1))  # (bs, HW, CH // 8)
        # phi path (second conv block)
        phi = self.phi(x)
        phi = torch.nn.functional.max_pool2d(phi, 2)
        phi = phi.view(batch_size, ch // 8, h * w // 4)  # (bs, CH // 8, HW)
        # attention path (theta and phi)
        attn = torch.bmm(theta, phi)  # (bs, HW, HW // 4)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        # g path (third conv block)
        g = self.g(x)
        g = torch.nn.functional.max_pool2d(g, 2)
        # (bs, HW // 4, CH)
        g = g.view(batch_size, ch, h * w // 4).permute((0, 2, 1))
        # attention map (g and attention path)
        attn_g = torch.bmm(attn, g)  # (bs, HW, CH)
        attn_g = attn_g.permute((0, 2, 1)).view(
            batch_size, ch, h, w)  # (bs, CH, H, W)
        # print(attn_g.shape)
        # attn_g = self.attn(attn_g)
        # print(attn_g.shape)
        return x + self.sigma_ratio * attn_g
