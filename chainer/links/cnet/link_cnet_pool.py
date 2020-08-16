from __future__ import absolute_import
import math
import numpy as np
import chainer
from chainer.links import CLink
from chainer.links.cnet.link_cnet_pool2D import CnetPool2D


class CnetPool(chainer.link.Chain, CLink):
    def __init__(self, pksize=3, pstride=2, ppad=0):
        super(CnetPool, self).__init__(
            pool=CnetPool2D(pksize, pstride, ppad)
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
        pout_w = (w + 2 * self.pool.pad - self.pool.kern) / self.pool.stride + 1
        pout_h = (h + 2 * self.pool.pad - self.pool.kern) / self.pool.stride + 1

        lname = name + '_' + 'layer_p'
        text.append('layer_t {} = {{}};'.format(lname))
        lpname = name + '_' + 'pool'
        text.append('pooling_layer_t {} = {{}};'.format(lpname))

        # text += 'float {}[{}] = {{}}'.format(lname + '_buf', c * self.pool.kern * self.pool.kern * w * h)
        # text += '{}.extra.value = {}'.format(lname, lname + '_buf')
        text = "\n".join(text) + '\n'
        ftext = "void {name}(float* input, float* output, float* buf){{\n"
        ftext += "  {lname}.batch = 1; \n"
        ftext += "  {lname}.input.size = {isize}; \n"
        ftext += "  {lname}.output.size = {osize}; \n"
        ftext += "  {lname}.extra.size = {esize}; \n"
        ftext += "  {lpname}.ic = {ic}; \n"
        ftext += "  {lpname}.iw = {iw}; \n"
        ftext += "  {lpname}.ih = {ih}; \n"
        ftext += "  {lpname}.oc = {oc}; \n"
        ftext += "  {lpname}.ow = {ow}; \n"
        ftext += "  {lpname}.oh = {oh}; \n"
        ftext += "  {lpname}.k = {k}; \n"
        ftext += "  {lpname}.s = {s}; \n"
        ftext += "  {lpname}.p = {p}; \n"
        ftext += "  {lname}.input.val = input; \n"
        ftext += "  {lname}.output.val = output; \n"
        ftext += "  {lname}.extra.val = buf; \n"
        ftext += "  max_pooling_layer_forward(&{lname}, &{lpname});\n}}\n\n"
        ftext = ftext.format(name=name, lname=lname, lpname=lpname, isize=(h * w), osize=(pout_h * pout_w),
                             esize=(c * self.pool.kern * self.pool.kern * w * h), ic=c, iw=w, ih=h, oc=c, ow=pout_w,
                             oh=pout_h, k=self.pool.kern, s=self.pool.stride, p=self.pool.pad)
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
