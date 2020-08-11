from __future__ import absolute_import
import math

import numpy as np
import chainer
from chainer.links import CLink
from chainer.links.cnet import link_cnet_convolution

class CnetConv(chainer.Chain, CLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0):
        super(CnetConv, self).__init__(
            conv=link_cnet_convolution.CnetConvolution2D(in_channels, out_channels, ksize=ksize, stride=stride, pad=pad)
        )
        self.cname = "l_cnet_conv"

    def __call__(self, h, test=False):
        h = self.conv(h)
        return h

    def generate_c(self, link_idx, inp_shape):
        #if not hasattr(self,'inp_shape'):
        #    raise Exception("no input shape found")
        #    return ""
        name = self.cname + str(link_idx)
        text = []
        c, w, h = inp_shape[1:4]
        sw, sh = self.conv.stride
        pw, ph = self.conv.pad
        kw, kh = self.conv.ksize
        out_w = (w + 2 * ph - kh) / sw + 1
        out_h = (h + 2 * ph - kh) / sw + 1

        # conv
        l = self.conv
        lname = name + '_' + 'layer'
        text += 'layer_t {}{{}};'.format(lname)
        text += '{}.batch = 1;'.format(lname)
        text += '{}.input.size = {};'.format(lname, np.prod(inp_shape))
        text += '{}.output.size = {};'.format(lname, np.prod(np.prod(l.b.data.shape)))
        text += '{}.weight.size = {};'.format(lname, np.prod(np.prod(l.W.data.shape)))
        text += '{}.bias.size = {};'.format(lname, np.prod(np.prod(l.b.data.shape)))
        text += '{}.extra.size = {};'.format(lname, (c * kh * kw * out_h * out_w))
        lcname = name + '_' + 'conv'
        text += 'conv_layer_t {}{{}};'.format(lcname)
        text += '{}.ic = {};'.format(lcname, c)
        text += '{}.iw = {};'.format(lcname, w)
        text += '{}.ih = {};'.format(lcname, h)
        text += '{}.oc = {};'.format(lcname, l.W.data.shape[0])
        text += '{}.ow = {};'.format(lcname, out_w)
        text += '{}.oh = {};'.format(lcname, out_h)
        text += '{}.k = {};'.format(lcname, kw)
        text += '{}.s = {};'.format(lcname, sw)
        text += '{}.p = {};'.format(lcname, pw)
        for p in l.params():
            pname = p.name
            if pname == 'W':
                num_f, n, kw, kh = p.data.shape
                bin_data = p.data.flatten()
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(bin_data), ','.join(map(str, bin_data)))
                text += [c_str]
                text += '{}.weight.value = {}'.format(lname, lname + '_' + pname)
            elif pname == 'b':
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(p.data), ','.join(map(str, p.data)))
                text += [c_str]
                text += '{}.bias.value = {}'.format(lname, lname + '_' + pname)
        text = "\n".join(text)+'\n'
        ftext = "void {name}(float* input, float* output){{\n"
        ftext += "{lname}.input.val = input; \n"
        ftext += "{lname}.output.val = output; \n"
        ftext += "{lname}.extra.val = buf; \n"
        ftext += "  conv_layer_forward({lname}, {lcname});\n}}\n\n"
        ftext = ftext.format(name=name, lname=lname, lcname=lcname)
        text += ftext

        return text

    def is_bin(self):
        return False

    def buf_mem(self, inp_shape):
        c, w, h = inp_shape[1:4]
        sw, sh = self.conv.stride
        pw, ph = self.conv.pad
        kw, kh = self.conv.ksize
        out_w = (w + 2 * ph - kh) / sw + 1
        out_h = (h + 2 * ph - kh) / sw + 1
        return c * kh * kw * out_h * out_w

    def temp_mem(self, inp_shape):
        w = np.prod(inp_shape)
        return w
