from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer.utils import conv
from ctypes import *
from six import moves
import numpy as np
import cupy
import time
import os
dllpath = '/home/monica/Documents/chainer-1.17.0-RAP/chainer/functions/cnet/libcnet.so'
lib = CDLL(dllpath, RTLD_GLOBAL)

# def _kern():
#     return cuda.elementwise(
#         'T x', 'T y',
#         'y = x >= 0 ? 1 : -1',
#         'binarize')

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class data_t(Structure):
    _fields_ = [("size", c_int),
                ("val", POINTER(c_float)),
                ("grad", POINTER(c_float))]

class conv_layer_t(Structure):
    _fields_ = [("ic", c_int),  # in height
                ("iw", c_int),  # in width
                ("ih", c_int),  # in channel
                ("oc", c_int),  # out height
                ("ow", c_int),  # out width
                ("oh", c_int),  # out channel
                ("k", c_int),   # kernel
                ("s", c_int),   # stride
                ("p", c_int)]    # padding

class layer(Structure):
    _fields_ = [("batch", c_int),
                ("input", data_t),
                ("output", data_t),
                ("weight", data_t),
                ("bias", data_t),
                ("extra", data_t)
                ]


forward_max_pooling_layer = lib.max_pooling_layer_forward
forward_max_pooling_layer.argtypes = [POINTER(layer), POINTER(conv_layer_t)]
backward_max_pooling_layer = lib.max_pooling_layer_backward
backward_max_pooling_layer.argtypes = [POINTER(layer), POINTER(conv_layer_t)]

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class CnetMaxPooling2DFunction(function.Function):

    def __init__(self, ksize, stride=None, pad=0, cover_all=True,
                 use_cudnn=True):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        self.cover_all = cover_all
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward_cpu(self, inputs):
        n, c, h, w = inputs[0].shape
        x = _as_mat(inputs[0])

        out_c = c
        out_w = (w + 2 * self.pw - self.kw) / self.sx + 1
        out_h = (h + 2 * self.ph - self.kh) / self.sy + 1
        output_size = out_h * out_w * out_c * n
        o = (c_float * output_size)()
        extra_size = c * self.kh * self.kw * out_w * out_h
        e = (c_float * extra_size)()

        l = layer()
        pool_p = conv_layer_t()
        l.batch = n
        l.input = data_t(c * h * w, cast(x.ctypes.data, POINTER(c_float)))
        # c_in = np.ctypeslib.as_ctypes(x)
        # l.input = data_t(n * c * h * w, c_in)
        l.output = data_t(out_h * out_w * out_c, o)
        l.bias = data_t(0)
        l.weight = data_t(0)
        l.extra = data_t(extra_size, e)
        pool_p.ih = h
        pool_p.iw = w
        pool_p.ic = c
        pool_p.oh = out_h
        pool_p.ow = out_w
        pool_p.oc = out_c
        pool_p.k = self.kh
        pool_p.s = self.sx
        pool_p.p = self.ph
        # print("ih"+ pool_p.ih + "iw"+ pool_p.iw + "ic"+ pool_p.ic + "oh"+ pool_p.oh + "ow"+ pool_p.ow + "oc"+ pool_p.oc + "k"+ pool_p.k +  "s"+ pool_p.s + "p"+ pool_p.p)
        # variables
        # l.h = h
        # l.w = w
        # l.c = c
        # l.n = out_c # filter sizeo = (c_float * output_size)()
        # l.groups = 1
        # l.stride = self.sx
        # l.size = kh
        # l.pad = self.ph
        # l.outputs = l.out_h * l.out_w * l.out_c
        # l.inputs = l.w * l.h * l.c

        # print("forward conv layer(c)")
        forward_max_pooling_layer(byref(l), byref(pool_p))
        # print("end conv layer")

        y = np.ctypeslib.as_array((c_float * (out_h * out_w * out_c * n)).from_address(addressof(o)))
        # print(y)
        y = np.reshape(y, (n, out_c, out_h, out_w))
        returnY = np.copy(y)
        # np.savetxt('cnet_conv_for_x.txt', x.flatten(), fmt='%f', delimiter=',')
        # np.savetxt('cnet_conv_for_W.txt', W.flatten(), fmt='%f', delimiter=',')
        # np.savetxt('cnet_conv_for_b.txt', b.flatten(), fmt='%f', delimiter=',')
        # time.sleep(0.1)
        # sec = time.time()
        # name = 'cnet_pool_for' + str(sec) + ".npy"
        # np.save(name, returnY.flatten())
        # print("pool forward output shape")
        # print(returnY.shape)
        return returnY,

    def forward_gpu(self, inputs):
        n, c, h, w = inputs[0].shape
        # x = _as_mat(inputs[0])

        nx = cupy.asnumpy(inputs[0])
        nx = nx.flatten()
        # print(np.shape(nx))
        # for i in range(n * c * h * w):
        #     if i % (28 * 28) is 0:
        #         print("\n"),
        #     print(nx[i]),
        # x = inputs[0]
        out_c = c
        out_w = (w + 2 * self.pw - self.kw) / self.sx + 1
        out_h = (h + 2 * self.ph - self.kh) / self.sy + 1
        output_size = out_h * out_w * out_c * n
        o = (c_float * output_size)()
        extra_size = c * self.kh * self.kw * out_w * out_h
        e = (c_float * extra_size)()


        l = layer()
        pool_p = conv_layer_t()
        l.batch = n
        l.input = data_t(c * h * w, cast(nx.ctypes.data, POINTER(c_float)))
        # print(n*c*h*w)
        l.output = data_t(out_h * out_w * out_c, o)
        l.bias = data_t(0)
        l.weight = data_t(0)
        l.extra = data_t(extra_size, e)
        pool_p.ih = h
        pool_p.iw = w
        pool_p.ic = c
        pool_p.oh = out_h
        pool_p.ow = out_w
        pool_p.oc = out_c
        pool_p.k = self.kh
        pool_p.s = self.sx
        pool_p.p = self.ph
        # string = "ih" + str(pool_p.ih) + "iw" + str(pool_p.iw) + "ic" + str(pool_p.ic) + "oh" + str(pool_p.oh) + "ow" + str(pool_p.ow) + "oc" + str(pool_p.oc) + "k" + str(pool_p.k) + "s" + str(pool_p.s) + "p" + str(pool_p.p)
        # print(string)
        # print("forward conv layer")
        forward_max_pooling_layer(byref(l), byref(pool_p))
        # print("end conv layer")

        y = np.ctypeslib.as_array((c_float * (out_h * out_w * out_c * n)).from_address(addressof(o)))
        y = np.reshape(y, (n, out_c, out_h, out_w))
        cy = cupy.asarray(y)
        # print(y.shape)
        # print("conv for")
        # print(cy)
        return cy,

    def backward_cpu(self, inputs, grad_outputs):
        # x, W = inputs[:2]
        # print("back_pool")
        # print(inputs[0].shape)
        # print(len(inputs[0]))
        x = _as_mat(inputs[0])
        n, c, h, w = inputs[0].shape
        on, out_c, out_w, out_h = grad_outputs[0].shape
        input_grad = (c_float * np.size(x))()
        extra_size = c * self.kh * self.kw * out_w * out_h
        e_grad = (c_float * extra_size)()
        e = (c_float * extra_size)()
        output_size = out_h * out_w * c * n
        o = (c_float * output_size)()
        # print(grad_outputs[0].shape)

        l = layer()
        pool_p = conv_layer_t()
        l.batch = n
        l.input = data_t(c * h * w, cast(x.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(out_h * out_w * out_c, o, cast(grad_outputs[0].ctypes.data, POINTER(c_float)))
        l.bias = data_t(0)
        l.weight = data_t(0)
        l.extra = data_t(extra_size, e, e_grad)
        pool_p.ih = h
        pool_p.iw = w
        pool_p.ic = c
        pool_p.oh = out_h
        pool_p.ow = out_w
        pool_p.oc = out_c
        pool_p.k = self.kh
        pool_p.s = self.sx
        pool_p.p = self.ph

        # print("backward conv layer (c)")
        backward_max_pooling_layer(byref(l), byref(pool_p))
        # print("end conv layer")
        gx = np.ctypeslib.as_array((c_float * np.size(x)).from_address(addressof(input_grad)))
        gx = gx.reshape(inputs[0].shape)
        rgx = np.copy(gx)
        # time.sleep(0.1)
        # sec = time.time()
        # namex = 'cnet_pool_back' + str(sec) + ".npy"
        # np.save(namex, rgx.flatten())
        # print(len(rgx))
        # print(rgx.shape)
        return rgx,

    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        nx = cupy.asnumpy(x)
        out_g = cupy.asnumpy(grad_outputs)
        # print(grad_outputs[0].shape)
        n, c, h, w = inputs[0].shape
        out_c = c
        out_w = (w + 2 * self.ph - self.kh) / self.sx + 1
        out_h = (h + 2 * self.ph - self.kh) / self.sx + 1

        T_batch = n
        input_grad = (c_float * np.size(x))()
        extra_size = c * self.kh * self.kh * out_w * out_h
        e_grad = (c_float * extra_size)()
        e = (c_float * extra_size)()
        output_size = out_h * out_w * c * n
        o = (c_float * output_size)()

        l = layer()
        pool_p = conv_layer_t()
        l.batch = T_batch
        l.input = data_t(c * h * w, cast(nx.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(out_h * out_w * out_c, o, cast(out_g.ctypes.data, POINTER(c_float)))
        l.bias = data_t(0)
        l.weight = data_t(0)
        l.extra = data_t(extra_size, e, e_grad)
        pool_p.ih = h
        pool_p.iw = w
        pool_p.ic = c
        pool_p.oh = out_h
        pool_p.ow = out_w
        pool_p.oc = out_c
        pool_p.k = self.kh
        pool_p.s = self.sx
        pool_p.p = self.ph

        # print("backward conv layer (g)")
        backward_max_pooling_layer(byref(l), byref(pool_p))
        # print("end conv layer")
        gx = np.ctypeslib.as_array((c_float * np.size(x)).from_address(addressof(input_grad)))
        gx = gx.reshape(inputs[0].shape)
        ygx = cupy.array(gx)

        return ygx,


def cnet_max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True,
                   use_cudnn=True):
    """Binary Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``..

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """

    return CnetMaxPooling2DFunction(ksize, stride, pad, cover_all, use_cudnn)(x)
