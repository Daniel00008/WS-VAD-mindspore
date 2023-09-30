import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
import ipdb

class _NonLocalBlockND(nn.Cell):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.SequentialCell(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.SequentialCell(self.g, max_pool_layer)
            self.phi = nn.SequentialCell(self.phi, max_pool_layer)
        
    def construct(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """       
        batch_size = x.shape[0]

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = ops.permute(g_x,(0, 2, 1))

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = ops.permute(theta_x,(0, 2, 1))
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = ops.matmul(theta_x, phi_x)
        N = f.shape[-1]
        f_div_C = f / N

        y = mindspore.ops.matmul(f_div_C, g_x)
        y = ops.permute(y,(0, 2, 1))# .contiguous()
        y = y.view(batch_size, self.inter_channels, *x.shape[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Cell):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.out_len = len_feature // 4
        self.conv_1 = nn.SequentialCell(
            nn.Conv1d(in_channels=len_feature, out_channels=self.out_len, kernel_size=3, pad_mode='pad',
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(self.out_len)
        )
        self.conv_2 = nn.SequentialCell(
            nn.Conv1d(in_channels=len_feature, out_channels=self.out_len, kernel_size=3,pad_mode='pad',
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(self.out_len)
        )
        self.conv_3 = nn.SequentialCell(
            nn.Conv1d(in_channels=len_feature, out_channels=self.out_len, kernel_size=3,pad_mode='pad',
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(self.out_len)
        )
        self.conv_4 = nn.SequentialCell(
            nn.Conv1d(in_channels=len_feature, out_channels=self.out_len, kernel_size=1,
                      stride=1, padding=0, has_bias=False),
            nn.ReLU(),
        )

        self.conv_5 = nn.SequentialCell(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3, pad_mode='pad',
                      stride=1, padding=1, has_bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
        )

        self.non_local = NONLocalBlock1D(self.out_len, sub_sample=False, bn_layer=True)

    def construct(self, x):
        # x: (B, T, F)
        out = ops.permute(x,(0, 2, 1))
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = ops.cat((out1, out2, out3), axis=1)
        out = self.conv_4(out)
        out = self.non_local(out)
        out = ops.cat((out_d, out), axis=1)
        out = self.conv_5(out)
        out = out + residual

        return out
