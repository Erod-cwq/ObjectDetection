import torch.nn as nn
from model.modules import Conv, Bottleneck, Concat, Detect
import torch
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, cfg='yolo.yaml'):
        super(Model, self).__init__()
        with open(cfg) as f:
            import yaml
            self.yaml = yaml.safe_load(f)
        self.layers = []
        self.model, self.save = parse_model(deepcopy(self.yaml))
        initialize_weights(self)

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def parse_model(cfg):
    ch = [3]  # input channel
    anchors = cfg['anchors']
    nc = cfg['nc']
    save, layers = [], []

    for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # for detection modules
            except NameError:
                pass

        if m in [Conv, Bottleneck]:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])

            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        m_.i, m_.f = i, f
        if i == 0:
            ch = []
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


if __name__ == '__main__':
    model = Model()
    model(torch.rand([1, 3, 640, 640]))
