"""Model definition.
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """Base ResNet model; see main.py for options.
    """

    def __init__(self, training_options):
        super().__init__()
        self.training_options = training_options
        self.register_buffer('input_offset', torch.Tensor(
                training_options['input_offset']))
        types = {
                # (a)ctivation, (b)atchnorm, (c)onvolution
                'a': HHReLU if training_options['robust_additions']
                    else nn.ReLU,
                'b': nn.BatchNorm2d,
                'c': nn.Conv2d,
        }

        pipe = []
        in_size, preprocess_fn, block_lens, block_fts, nclass = (
                training_options['arch'])
        pipe.append(preprocess_fn(block_fts[0]))
        ft = block_fts[0]
        for i, (blen, bft) in enumerate(zip(block_lens, block_fts)):
            pipe.append(_SuperBlock(types, ft, bft, blen,
                    stride=1 if i == 0 else 2))
            ft = bft
        pipe.append(types['b'](ft))
        pipe.append(types['a']())
        pipe.append(nn.AvgPool2d(in_size // 2 ** (len(block_lens)-1)))
        pipe.append(types['c'](ft, nclass, 1))
        self.pipe = nn.Sequential(*pipe)

        # Standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight is not None and nn.init.kaiming_uniform_(m.weight.data)
                m.bias is not None and m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Trick - all resnet blocks do nothing, at first.  Basically initializes
        # training with a very shallow network.
        # See arXiv:1812.01187
        for m in self.modules():
            if isinstance(m, _PreActBlock):
                m.main[-1].weight.data.zero_()


    def forward(self, x):
        x = x - self.input_offset[0].view(1, -1, 1, 1)
        x /= self.input_offset[1].view(1, -1, 1, 1)
        x = self.pipe(x)
        assert x.size(2) == 1 and x.size(3) == 1, x.size()
        return x[:, :, 0, 0]



class _SuperBlock(nn.Sequential):
    def __init__(self, types, ft_in, ft_out, nblock, stride):
        layers = []
        for i in range(nblock):
            block_in = ft_in if i == 0 else ft_out
            if i != 0:
                stride = 1
            layers.append(_PreActBlock(types, block_in, ft_out, stride))
        super().__init__(*layers)



class _PreActBlock(nn.Module):
    """Mostly from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
    """
    def __init__(self, types, nf_in, nf, stride):
        super().__init__()
        self.pre = nn.Sequential(
                types['b'](nf_in),
                types['a'](),
        )

        self.main = nn.Sequential(
                types['c'](nf_in, nf, 3, padding=1, stride=stride, bias=False),
                types['b'](nf),
                types['a'](),
                types['c'](nf, nf, 3, padding=1, bias=False),
        )
        if stride != 1 or nf_in != nf:
            self.shortcut = types['c'](nf_in, nf, 3, padding=1, stride=stride,
                    bias=False)
        else:
            self.shortcut = None
    def forward(self, x):
        pre = self.pre(x)
        main = self.main(pre)
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(pre)
        return main + shortcut



class HHReLU(nn.Module):
    """Like ReLU, but with a square bowl s.t. derivative is smooth around zero.
    """
    DELTA = 0.5  # Natural choice for imitating ReLU (?)
    def __init__(self):
        super().__init__()
        self.register_buffer('d', torch.tensor(self.DELTA, dtype=torch.float))

    def forward(self, x):
        return HHReLU._forward2.apply(x, self.d)

    class _forward2(torch.autograd.Function):
        """More efficient due to usage of inplace operators, which aren't
        allowed through autograd.
        """
        @staticmethod
        def forward(ctx, x, d):
            ctx.save_for_backward(x, d)

            x = torch.nn.functional.relu(x)
            q = (x.detach() > d).float()

            # Cover linear region
            x.add_(q, alpha=-0.5 * d)

            # Hmm... another temporary for multiplying x to get bowl region
            nq = q.neg().add_(1. / (2 * d))
            nq.mul_(x)
            nq.add_(q)
            x.mul_(nq)
            return x
        @staticmethod
        def backward(ctx, grad_output):
            x, d = ctx.saved_tensors

            g = x.clamp(0, d)
            g.div_(d).mul_(grad_output)
            return g, None

