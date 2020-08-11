from __future__ import absolute_import
import math

import numpy as np
import chainer

from chainer.links import CLink
from chainer.links.cnet.link_cnet_pool2D import CnetPool2D
from chainer.utils import binary_util as bu

class CnetPool(chainer.Chain, CLink):
    def __init__(self, in_channels, out_channels, pksize=3, pstride=2, ppad=0):
        super(CnetPool, self).__init__(
            pool=CnetPool2D(pksize,pstride,ppad)
        )
        self.cname = "l_cnet_pool"

    def __call__(self, h, test=False):
        h = self.pool(h)
        return h

    def generate_c(self, link_idx, inp_shape):
        name = self.cname + str(link_idx)
        text = []
        c, w, h = inp_shape[1:4]
        # pool
        pout_w = (w + 2 * self.pool.pw - self.pool.kw) / self.pool.sx + 1
        pout_h = (h + 2 * self.pool.ph - self.pool.kh) / self.pool.sy + 1

        lname = name + '_' + 'layer_p'
        text += 'layer_t {}{{}};'.format(lname)
        text += '{}.batch = 1;'.format(lname)
        text += '{}.input.size = {};'.format(lname, (h * w))
        text += '{}.output.size = {};'.format(lname, (pout_h * pout_w))
        text += '{}.extra.size = {};'.format(lname, (c * self.pool.kern * self.pool.kern * w * h))
        lpname = name + '_' + 'pool'
        text += 'conv_layer_t {}{{}};'.format(lpname)
        text += '{}.ic = {};'.format(lpname, c)
        text += '{}.iw = {};'.format(lpname, w)
        text += '{}.ih = {};'.format(lpname, h)
        text += '{}.oc = {};'.format(lpname, c)
        text += '{}.ow = {};'.format(lpname, pout_w)
        text += '{}.oh = {};'.format(lpname, pout_h)
        text += '{}.k = {};'.format(lpname, self.pool.kern)
        text += '{}.s = {};'.format(lpname, self.pool.stride)
        text += '{}.p = {};'.format(lpname, self.pool.pad)

        # text += 'float {}[{}] = {{}}'.format(lname + '_buf', c * self.pool.kern * self.pool.kern * w * h)
        # text += '{}.extra.value = {}'.format(lname, lname + '_buf')
        text = "\n".join(text) + '\n'
        ftext = "void {name}(float* input, float* output){{\n"
        # TODO extra buffer
        ftext += "{lname}.input.val = input; \n"
        ftext += "{lname}.output.val = output; \n"
        ftext += "{lname}.extra.val = buf; \n"
        ftext += "  max_pooling_layer_forward({lname}, {lpname});\n"
        ftext += "}}\n\n"
        ftext = ftext.format(name=name, lname=lname, lpname=lpname)
        text += ftext
        return text

    def is_bin(self):
        return False

    def buf_mem(self, inp_shape):
        c, w, h = inp_shape[1:4]
        w = c * self.pool.kern * self.pool.kern * w * h
        return w

    def temp_mem(self, inp_shape):
        w = np.prod(inp_shape)
        return w
