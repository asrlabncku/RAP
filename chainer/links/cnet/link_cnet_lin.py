from __future__ import absolute_import
import math
import numpy as np
import chainer
from chainer.links import CLink
from chainer.links.cnet import link_cnet_linear


class CnetLin(chainer.link.Chain, CLink):
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
        text.append('layer_t {} = {{}};'.format(lname))
        for p in l.params():
            pname=p.name
            if pname == 'W':
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
        ftext += "  {lname}.input.val = input; \n"
        ftext += "  {lname}.output.val = output; \n"
        ftext += "  {lname}.weight.val = {wname}; \n"
        ftext += "  {lname}.bias.val = {bname}; \n"
        ftext += "  fc_layer_forward(&{lname}); \n}}\n\n"
        ftext = ftext.format(name=name, lname=lname, isize=np.prod(inp_shape), osize=np.prod(l.b.data.shape),
                             wsize=np.prod(l.W.data.shape), bsize=np.prod(l.b.data.shape), wname=lname + '_W',
                             bname=lname + '_b')
        text += ftext

        return text

    def is_bin(self):
        return False

    def buf_mem(self, inp_shape):
        return 0

    def temp_mem(self, inp_shape):
        w = np.prod(inp_shape)
        return w
