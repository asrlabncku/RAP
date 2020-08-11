from __future__ import absolute_import
import math

import chainer
import numpy as np

from chainer.links import CLink
from chainer.links.cnet import link_cnet_linear

class CnetLin(chainer.Chain, CLink):
    def __init__(self, in_channels, out_channels):
        super(CnetLin, self).__init__(
            bl=link_cnet_linear.CnetLinear(in_channels, out_channels)
        )
        self.cname = "l_cnet_lin"
        
    def __call__(self, h, test=False):
        h = self.bl(h)
        return h

    def generate_c(self, link_idx, inp_shape):
        name = self.cname + str(link_idx)
        text = []

        # BinaryLinear bl
        l = self.bl
        lName = l.name
        lname=name+'_'+ 'layer'
        text += 'layer_t {}{{}};'.format(lname)
        text += '{}.batch = 1;'.format(lname)
        text += '{}.input.size = {};'.format(lname, np.prod(inp_shape))
        text += '{}.output.size = {};'.format(lname, np.prod(np.prod(l.b.data.shape)))
        text += '{}.weight.size = {};'.format(lname, np.prod(np.prod(l.W.data.shape)))
        text += '{}.bias.size = {};'.format(lname, np.prod(np.prod(l.b.data.shape)))
        for p in l.params():
            pname=p.name
            if pname == 'W':
                bin_data = p.data.flatten()
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(bin_data), ','.join(map(str, bin_data)))
                text += [c_str]
                text += '{}.weight.value = {};'.format(lname, lname + '_' + pname)
            elif pname == 'b':
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(p.data), ','.join(map(str, p.data)))
                text += [c_str]
                text += '{}.bias.value = {};'.format(lname, lname + '_' + pname)

        text = "\n".join(text)+'\n'
        ftext = "void {name}(float* input, float* output){{\n"
        ftext += "{lname}.input.val = input; \n"
        ftext += "{lname}.output.val = output; \n"
        ftext += "  fc_layer_forward({lname}); \n}}\n\n"
        ftext = ftext.format(name=name, lname=lname)
        text += ftext

        return text

    def is_bin(self):
        return False

    def buf_mem(self, inp_shape):
        return 0

    def temp_mem(self, inp_shape):
        w = np.prod(inp_shape)
        return w
