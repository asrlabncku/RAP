import numpy
import chainer
import chainer.utils

# from chainer import links
import chainer

class BinaryLink(object):
    def __call__(self, x):
        raise NotImplementedError()

    def has_multiple_weights(self):
        return False

    def from_dict(self, dict):
        for attr, value in dict.iteritems():
            setattr(self, attr, value)

    def to_dict(self):
        dict = {}
        for attr, value in self.__dict__.iteritems():
            dict[attr] = value
        return dict

    def to_chainer_args(self):
        dict = {}
        for attr, value in self.__dict__.iteritems():
            if attr[0] == "_":
                pass
            else:
                dict[attr] = value
        return dict

    def to_link(self):
        raise NotImplementedError()

    def dump(self):
        print("Link: {}".format(self._link))
        for attr, value in self.__dict__.iteritems():
            print("    {}: {}".format(attr, value))

class BinaryConvBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0):
        self._link = "BinaryConvBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.BinaryConvBNBST(**args)

class ConvBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0):
        self._link = "ConvBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def to_link(self):
        args = self.to_chainer_args()
        return chainer.links.ConvBNBST(**args)

class BinaryConvPoolBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, pksize=3, pstride=2, ppad=0):
        self._link = "BinaryConvPoolBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.pksize = pksize
        self.pstride = pstride
        self.ppad = ppad

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.BinaryConvPoolBNBST(**args)
class ConvPoolBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, pksize=3, pstride=2, ppad=0):
        self._link = "ConvPoolBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.pksize = pksize
        self.pstride = pstride
        self.ppad = ppad

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        #print('test enter to_link()')
        return chainer.links.ConvPoolBNBST(**args)

class CnetPool(BinaryLink):
    def __init__(self, pksize=3, pstride=2, ppad=0):
        self._link = "CnetPool"
        self.pksize = pksize
        self.pstride = pstride
        self.ppad = ppad

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        #print('test enter to_link()')
        return chainer.links.CnetPool(**args)

class CnetConv(BinaryLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0):
        self._link = "CnetConv"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        #print('test enter to_link()')
        return chainer.links.CnetConv(**args)

class BinaryLinearBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels):
        self._link = "BinaryLinearBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.BinaryLinearBNBST(**args)

class BinaryLinearBNSoftmax(BinaryLink):
    def __init__(self, in_channels, out_channels):
        self._link = "BinaryLinearBNSoftmax"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.BinaryLinearBNSoftmax(**args)

class CnetLin(BinaryLink):
    def __init__(self, in_channels, out_channels):
        self._link = "CnetLin"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.CnetLin(**args)


class BinaryLinearSoftmax(BinaryLink):
    def __init__(self, in_channels, out_channels):
        self._link = "BinaryLinearSoftmax"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.BinaryLinearSoftmax(**args)

class LinearBNBST(BinaryLink):
    def __init__(self, in_channels, out_channels):
        self._link = "LinearBNBST"
        self.in_channels = in_channels
        self.out_channels = out_channels

    def to_link(self):
        # TODO Support Weight Initializer
        args = self.to_chainer_args()
        return chainer.links.LinearBNBST(**args)