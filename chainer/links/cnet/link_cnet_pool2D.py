from __future__ import absolute_import
from chainer import link

from chainer.functions.cnet import function_cnet_maxpool

class CnetPool2D(link.Link):
    def __init__(self, kern=3, stride=2, pad=0):
        self.kern = kern
        self.stride = stride
        self.pad = pad
        super(CnetPool2D, self).__init__()
        self.cname = "l_cnet_pool2D"

    def __call__(self, x):
        return function_cnet_maxpool.cnet_max_pooling_2d(x, ksize=self.kern, stride=self.stride, pad=self.pad, cover_all=False)
