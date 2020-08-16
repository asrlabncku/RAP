import math
import chainer
from chainer import cuda
from chainer.variable import Variable
# from chainer.functions import clipped_relu as f_clipped_relu
# from chainer.functions import crelu as f_crelu
# from chainer.functions import elu as f_elu
# from chainer.functions import hard_sigmoid as f_hard_sigmoid
# from chainer.functions import leaky_relu as f_leaky_relu
# from chainer.functions import log_softmax as f_log_softmax
# from chainer.functions import maxout as f_maxout
# from chainer.functions import relu as f_relu
# from chainer.functions import sigmoid as f_sigmoid
# from chainer.functions import softmax as f_softmax
# from chainer.functions import softplus as f_softplus
# from chainer.functions import tanh as f_tanh
# from chainer.functions import dropout as f_dropout
# from chainer.functions import gaussian as f_gaussian
# from chainer.functions import average_pooling_2d as f_average_pooling_2d
# from chainer.functions import max_pooling_2d as f_max_pooling_2d
# from chainer.functions import spatial_pyramid_pooling_2d as f_spatial_pyramid_pooling_2d
# from chainer.functions import unpooling_2d as f_unpooling_2d
# from chainer.functions import reshape as f_reshape
# from chainer.functions import softmax_cross_entropy as f_softmax_cross_entropy


class Function(object):

    def __call__(self, x):
        raise NotImplementedError()

    def from_dict(self, dict):
        for attr, value in dict.iteritems():
            setattr(self, attr, value)

    def to_dict(self):
        dict = {}
        for attr, value in self.__dict__.iteritems():
            dict[attr] = value
        return dict

class Activation(object):
    def __init__(self, nonlinearity="relu"):
        self.nonlinearity = nonlinearity

    def to_function(self):
        if self.nonlinearity.lower() == "clipped_relu":
            return clipped_relu()
        if self.nonlinearity.lower() == "crelu":
            return crelu()
        if self.nonlinearity.lower() == "elu":
            return elu()
        if self.nonlinearity.lower() == "hard_sigmoid":
            return hard_sigmoid()
        if self.nonlinearity.lower() == "leaky_relu":
            return leaky_relu()
        if self.nonlinearity.lower() == "relu":
            return relu()
        if self.nonlinearity.lower() == "sigmoid":
            return sigmoid()
        if self.nonlinearity.lower() == "softmax":
            return softmax()
        if self.nonlinearity.lower() == "softplus":
            return softplus()
        if self.nonlinearity.lower() == "tanh":
            return tanh()
        if self.nonlinearity.lower() == "bst":
            return bst()
        raise NotImplementedError()


from chainer.functions.eBNN import function_bst
class bst(Function):
    def __init__(self):
        self._function = "bst"

    def __call__(self, x):
        return function_bst.bst(x)

class clipped_relu(Function):
    def __init__(self, z=20.0):
        self._function = "clipped_relu"
        self.z = z

    def __call__(self, x):
        return chainer.functions.clipped_relu(x, self.z)

class crelu(Function):
    def __init__(self, axis=1):
        self._function = "crelu"
        self.axis = axis

    def __call__(self, x):
        return chainer.functions.crelu(x, self.axis)

class elu(Function):
    def __init__(self, alpha=1.0):
        self._function = "elu"
        self.alpha = alpha

    def __call__(self, x):
        return chainer.functions.elu(x, self.alpha)

class hard_sigmoid(Function):
    def __init__(self):
        self._function = "hard_sigmoid"
        pass

    def __call__(self, x):
        return chainer.functions.hard_sigmoid(x)

class leaky_relu(Function):
    def __init__(self, slope=0.2):
        self._function = "leaky_relu"
        self.slope = slope

    def __call__(self, x):
        return chainer.functions.leaky_relu(x, self.slope)

class log_softmax(Function):
    def __init__(self, use_cudnn=True):
        self._function = "log_softmax"
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.log_softmax(x, self.use_cudnn)

class maxout(Function):
    def __init__(self, pool_size, axis=1):
        self._function = "maxout"
        self.pool_size = pool_size
        self.axis = axis

    def __call__(self, x):
        return chainer.functions.maxout(x, self.pool_size, self.axis)

class relu(Function):
    def __init__(self, use_cudnn=True):
        self._function = "relu"
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.relu(x, self.use_cudnn)

class sigmoid(Function):
    def __init__(self, use_cudnn=True):
        self._function = "sigmoid"
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.sigmoid(x, self.use_cudnn)

class softmax(Function):
    def __init__(self, use_cudnn=True):
        self._function = "softmax"
        self.use_cudnn = use_cudnn
        pass
    def __call__(self, x):
        return chainer.functions.softmax(x, self.use_cudnn)

class softplus(Function):
    def __init__(self, use_cudnn=True):
        self._function = "softplus"
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.softplus(x, self.use_cudnn)

class tanh(Function):
    def __init__(self, use_cudnn=True):
        self._function = "tanh"
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.tanh(x, self.use_cudnn)

class dropout_comm_test(Function):
    def __init__(self, ratio=0.5):
        self._function = "dropout_comm_test"
        self.ratio = ratio

    def __call__(self, x, train=True):
        if not train:
            return chainer.functions.dropout(x, self.ratio, True)
        return x
    
class dropout_comm_train(Function):
    def __init__(self, ratio=0.5):
        self._function = "dropout_comm_train"
        self.ratio = ratio

    def __call__(self, x, train=True):
        if train:
            return chainer.functions.dropout(x, self.ratio, True)
        return x

class dropout(Function):
    def __init__(self, ratio=0.5):
        self._function = "dropout"
        self.ratio = ratio

    def __call__(self, x, train=True):
        return chainer.functions.dropout(x, self.ratio, train)

class gaussian_noise(Function):
    def __init__(self, std=0.3):
        self._function = "gaussian_noise"
        self.std = std

    def __call__(self, x):
        xp = cuda.get_array_module(x.data)
        ln_var = math.log(self.std ** 2)
        noise = chainer.functions.gaussian(Variable(xp.zeros_like(x.data)), Variable(xp.full_like(x.data, ln_var)))
        return x + noise

class average_pooling_2d(Function):
    def __init__(self, ksize, stride=None, pad=0, use_cudnn=True):
        self._function = "average_pooling_2d"
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.average_pooling_2d(x, self.ksize, self.stride, self.pad, self.use_cudnn)

class max_pooling_2d(Function):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True, use_cudnn=True):
        self._function = "max_pooling_2d"
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.max_pooling_2d(x, self.ksize, self.stride, self.pad, self.cover_all, self.use_cudnn)

class spatial_pyramid_pooling_2d(Function):
    def __init__(self, pyramid_height, pooling_class, use_cudnn=True):
        self._function = "spatial_pyramid_pooling_2d"
        self.pyramid_height = pyramid_height
        self.pooling_class = pooling_class
        self.use_cudnn = use_cudnn

    def __call__(self, x):
        return chainer.functions.spatial_pyramid_pooling_2d(x, self.pyramid_height, self.pooling_class, self.use_cudnn)

class unpooling_2d(Function):
    def __init__(self, ksize, stride=None, pad=0, outsize=None, cover_all=True):
        self._function = "unpooling_2d"
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.outsize = outsize
        self.cover_all = cover_all

    def __call__(self, x):
        return chainer.functions.unpooling_2d(x, self.ksize, self.stride, self.pad, self.outsize, self.cover_all)

class reshape(Function):
    def __init__(self, shape):
        self._function = "reshape"
        self.shape = shape

    def __call__(self, x):
        return chainer.functions.reshape(x, self.shape)

class reshape_1d(Function):
    def __init__(self):
        self._function = "reshape_1d"

    def __call__(self, x):
        batchsize = x.data.shape[0]
        return chainer.functions.reshape(x, (batchsize, -1))
    
class softmax_cross_entropy(Function):
    def __init__(self):
        self._function = "softmax_cross_entropy"

    def __call__(self, x, t):
        return chainer.functions.softmax_cross_entropy(x,t)
