import torch
import torch.nn as nn
from network.models import custom_layers as ops
from network.models.basic_model import BasicModel


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel, channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel, channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel, channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction) * 3, channel, 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ops.ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = ops.BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out


class DrlnBlock(nn.Module):
    def __init__(self, chs):
        super(DrlnBlock, self).__init__()
        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.c1 = ops.BasicBlock(chs * 2, chs, 3, 1, 1)
        self.c2 = ops.BasicBlock(chs * 3, chs, 3, 1, 1)
        self.c3 = ops.BasicBlock(chs * 4, chs, 3, 1, 1)

    def forward(self, x):
        b1 = self.b1(x)
        c1 = torch.cat([x, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3 + x


class Drln(BasicModel):
    def __init__(self, num_drln_blocks: int = 2, scale: int = 4, num_channels: int = 1):
        self.model_name = f"Drln_{num_drln_blocks}"
        super(Drln, self).__init__(self.model_name)
        self.scale = scale
        chs = 64
        self.head = nn.Conv2d(num_channels, chs, 3, 1, 1)
        self.upsample = ops.UpsampleBlock(chs, self.scale, multi_scale=False)
        # self.convert = ops.ConvertBlock(chs, chs, 20)
        self.tail = nn.Conv2d(chs, num_channels, 3, 1, 1)
        drln_blocks = [DrlnBlock(chs) for _ in range(num_drln_blocks)]
        self.drln_blocks = nn.Sequential(*drln_blocks)

    @staticmethod
    def reshape_to_model_output(low_res, high_res, device):
        """input shape is (N,doppler_points,range_points)"""
        if low_res.ndim == 4 and high_res.ndim == 4:
            return low_res.to(device), high_res.to(device)
        batch_size = low_res.shape[0]
        X = low_res.reshape(batch_size, 1, low_res.shape[2], low_res.shape[3])
        y = high_res.reshape(batch_size, 1, high_res.shape[2], high_res.shape[3])
        return X.to(device), y.to(device)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.head(x)
        c0 = o0 = x
        o0 = self.drln_blocks(o0)
        b_out = o0 + c0
        out = self.upsample(b_out, scale=self.scale)
        out = self.tail(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") >= 0 or name.find("upsample") >= 0:
                        print("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
