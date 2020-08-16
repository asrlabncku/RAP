from __future__ import absolute_import
import math
import chainer
import numpy as np
from chainer.chain_RAP import ChainRAP
from chainer.links import CLink
from chainer.links.cnet import link_cnet_convolution

class CnetConv(chainer.link.Chain, CLink):
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
        kw, kh = self.conv.W.data.shape[2:4]
        out_w = (w + 2 * ph - kh) / sw + 1
        out_h = (h + 2 * ph - kh) / sw + 1

        # conv
        l = self.conv
        lname = name + '_' + 'layer'
        text.append('layer_t {} = {{}};'.format(lname))
        isize = np.prod(inp_shape)
        osize = np.prod(l.b.data.shape)
        wsize = np.prod(l.W.data.shape)
        bsize = np.prod(l.b.data.shape)
        esize = c * kh * kw * out_h * out_w
        lcname = name + '_' + 'conv'
        text.append('conv_layer_t {} = {{}};'.format(lcname))
        oc = l.W.data.shape[0]
        for p in l.params():
            pname = p.name
            if pname == 'W':
                num_f, n, kw, kh = p.data.shape
                bin_data = p.data.flatten()
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(bin_data), ','.join(map(str, bin_data)))
                text += [c_str]
            elif pname == 'b':
                c_str = 'float {}[{}] = {{{}}};'.format(lname + '_' + pname, len(p.data), ','.join(map(str, p.data)))
                text += [c_str]
        text = "\n".join(text)+'\n'
        ftext = "void {name}(float* input, float* output, float* buf){{\n"
        ftext += "  {lname}.batch = 1; \n"
        ftext += "  {lname}.input.size = {isize}; \n"
        ftext += "  {lname}.output.size = {osize}; \n"
        ftext += "  {lname}.weight.size = {wsize}; \n"
        ftext += "  {lname}.bias.size = {bsize}; \n"
        ftext += "  {lname}.extra.size = {esize}; \n"
        ftext += "  {lcname}.ic = {ic}; \n"
        ftext += "  {lcname}.iw = {iw}; \n"
        ftext += "  {lcname}.ih = {ih}; \n"
        ftext += "  {lcname}.oc = {oc}; \n"
        ftext += "  {lcname}.ow = {ow}; \n"
        ftext += "  {lcname}.oh = {oh}; \n"
        ftext += "  {lcname}.k = {k}; \n"
        ftext += "  {lcname}.s = {s}; \n"
        ftext += "  {lcname}.p = {p}; \n"
        ftext += "  {lname}.input.val = input; \n"
        ftext += "  {lname}.output.val = output; \n"
        ftext += "  {lname}.weight.val = {wname}; \n"
        ftext += "  {lname}.bias.val = {bname}; \n"
        ftext += "  {lname}.extra.val = buf; \n"
        ftext += "  conv_layer_forward(&{lname}, &{lcname});\n}}\n\n"
        ftext = ftext.format(name=name, lname=lname, lcname=lcname, isize=isize, osize=osize, wsize=wsize, bsize=bsize,
                             esize=esize, ic=c, iw=w, ih=h, oc=oc, ow=out_w, oh=out_h, k=kw, s=sw, p=pw,
                             wname=lname + '_W', bname=lname + '_b')
        text += ftext

        return text

    def is_bin(self):
        return False

    def buf_mem(self, inp_shape):
        c, w, h = inp_shape[1:4]
        sw, sh = self.conv.stride
        pw, ph = self.conv.pad
        kw, kh = self.conv.W.data.shape[2:4]
        out_w = (w + 2 * ph - kh) / sw + 1
        out_h = (h + 2 * ph - kh) / sw + 1
        return c * kh * kw * out_h * out_w

    def temp_mem(self, inp_shape):
        w = np.prod(inp_shape)
        return w
