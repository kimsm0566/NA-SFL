import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing.nlp_tools import create_embedding_matrix


def make_layers(layer_type, **args):
    return getattr(sys.modules[__name__], layer_type)(**args)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out=None):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.input_dim is None:
            self.linear = nn.LazyLinear(self.output_dim)
        else:
            self.linear = nn.Linear(self.input_dim, self.output_dim)
        self.drop_out = nn.Dropout(drop_out) if drop_out else None

    def forward(self, x):
        out = self.linear(x)
        if self.drop_out:
            out = self.drop_out(out)
        return out


class Convolutional(nn.Module):
    def __init__(self, in_channel_num=None, out_channel_num=None, kernel_size=None, padding=1, stride=1,
                 bias=True, pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0,
                 non_linear='relu'):
        super(Convolutional, self).__init__()
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        # set up layers
        self.conv = nn.Conv2d(self.in_channel_num, self.out_channel_num,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              stride=self.stride,
                              bias=self.bias)
        self.bn = nn.BatchNorm2d(self.out_channel_num)
        if non_linear == 'relu':
            self.nl = nn.ReLU(inplace=True)
        elif non_linear == 'HSwish':
            self.nl = HSwish(inplace=True)
        else:
            raise Exception("Unrecognized non-linear layer name: {}".format(non_linear))
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nl(out)
        if self.pool:
            out = self.pool(out)
        return out


class ResBasicBlock(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, stride,
                 pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_num)
        self.conv3 = None
        self.bn3 = None
        if stride != 1 or in_channel_num != out_channel_num:
            self.conv3 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 and self.bn3:
            identity = self.conv3(identity)
            identity = self.bn3(identity)
        out += identity
        out = self.relu2(out)
        if self.pool:
            out = self.pool(out)
        return out


class ResBottleneck(nn.Module):
    def __init__(self,  in_channel_num, out_channel_num, stride,
                 pool_type=None, pool_kernel_size=None, pool_stride=None, pool_padding=0):
        expansion = 4
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channel_num, out_channel_num * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel_num * expansion)
        self.conv4 = None
        self.bn4 = None
        if stride != 1 or in_channel_num != out_channel_num * expansion:
            self.conv4 = nn.Conv2d(in_channel_num, out_channel_num * expansion, kernel_size=1, stride=stride, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channel_num * expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.stride = stride
        if pool_type == 'maxpool':
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        elif pool_type == 'avgpool':
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        else:
            self.pool = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.conv4 and self.bn4:
            identity = self.conv4(identity)
            identity = self.bn4(identity)
        out += identity
        out = self.relu3(out)
        if self.pool:
            out = self.pool(out)
        return out


class MobileBottleneck(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, kernel_size, stride, expansion,
                 non_linear='relu', use_se=False):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        padding = (kernel_size - 1) // 2
        self.use_res_connect = stride == 1 and in_channel_num == out_channel_num
        self.conv1 = nn.Conv2d(in_channel_num, expansion, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion)
        if non_linear == 'relu':
            self.nl1 = nn.ReLU(inplace=True)
            self.nl2 = nn.ReLU(inplace=True)
        elif non_linear == 'HSwish':
            self.nl1 = HSwish(inplace=True)
            self.nl2 = HSwish(inplace=True)
        else:
            raise Exception("Unrecognized non-linear layer name: {}".format(non_linear))
        self.conv2 = nn.Conv2d(expansion, expansion, kernel_size, stride, padding, groups=expansion, bias=False)
        self.se = SEModule(expansion) if use_se else None
        self.bn2 = nn.BatchNorm2d(expansion)
        self.conv3 = nn.Conv2d(expansion, out_channel_num, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel_num)

    def forward(self, x):
        identity = x if self.use_res_connect else 0
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.nl1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.nl2(out)
        out = self.conv3(out)
        # out = self.bn3(out)
        out += identity
        return out


class MobileLastConv(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, drop_out=None):
        super(MobileLastConv, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=1, padding=0)
        self.nl = HSwish(inplace=True)
        self.drop_out = nn.Dropout(p=drop_out) if drop_out else None

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = self.nl(out)
        if self.drop_out is not None:
            out = self.drop_out(out)
        return out


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class FCNHead(nn.Module):
    def __init__(self, in_channel_num, out_channel_num):
        super(FCNHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel_num, in_channel_num, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channel_num, in_channel_num, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)  # Transposed Convolution
        self.classifier = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.classifier(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channel_num, out_channel_num):
        super(ASPP, self).__init__()
        # Dilation rates can be modified
        self.dilation_rates = [1, 6, 12, 18]
        self.aspp_blocks = nn.ModuleList()
        for rate in self.dilation_rates:
            self.aspp_blocks.append(AtrousConvolution(in_channel_num, out_channel_num, rate))
        self.output_conv = nn.Conv2d(len(self.dilation_rates) * out_channel_num, out_channel_num, kernel_size=1)

    def forward(self, x):
        aspp_outputs = [block(x) for block in self.aspp_blocks]
        aspp_output = torch.cat(aspp_outputs, dim=1)
        return self.output_conv(aspp_output)


class AtrousConvolution(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, dilation_rate):
        super(AtrousConvolution, self).__init__()
        self.atrous_conv = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3,
                                     stride=1, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return self.atrous_conv(x)


class Interpolate(nn.Module):
    def __init__(self, input_image_height, input_image_width):
        super(Interpolate, self).__init__()
        self.input_image_height = input_image_height
        self.input_image_width = input_image_width

    def forward(self, x):
        return F.interpolate(
            x, size=(self.input_image_height, self.input_image_width), mode='bilinear', align_corners=False)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab, emb_dim, drop_out=None, pretrained=False):
        super(TransformerEmbedding, self).__init__()
        if pretrained:
            embedding_matrix = create_embedding_matrix(vocab, emb_dim)
            self.tok_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.tok_embedding = nn.Embedding(len(vocab), emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, drop_out=drop_out)
        self.emb_dim = emb_dim

    def forward(self, x):
        x = self.tok_embedding(x) * math.sqrt(self.emb_dim)
        x = self.pos_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000, drop_out=None):
        super().__init__()
        self.drop_out = nn.Dropout(drop_out) if drop_out else None
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float()
            * (-math.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        if self.drop_out:
            x = self.drop_out(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, hid_dim, drop_out):
        super(TransformerEncoderLayer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, hid_dim, drop_out, batch_first=True)

    def forward(self, x):
        return self.encoder_layer(x)
