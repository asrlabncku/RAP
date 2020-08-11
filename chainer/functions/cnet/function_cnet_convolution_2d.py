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
dllpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libcnet.so'))
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
                ("extra", data_t)]


forward_convolutional_layer = lib.conv_layer_forward
forward_convolutional_layer.argtypes = [POINTER(layer), POINTER(conv_layer_t)]
backward_convolutional_layer = lib.conv_layer_backward
backward_convolutional_layer.argtypes = [POINTER(layer), POINTER(conv_layer_t)]

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class CnetConvolution2DFunction(function.Function):

    def __init__(self, stride=1, pad=0, use_cudnn=True):
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype == np.float32,
            w_type.dtype == np.float32,
            x_type.ndim == 4,
            w_type.ndim == 4,
            x_type.shape[1] == w_type.shape[1],
        )

        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == np.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        # np.savetxt('cnet_conv_for_in_ori.txt', inputs[0].flatten(), fmt='%f', delimiter=',')
        n, c, h, w = inputs[0].shape
        # print("forward input shape " + str(inputs[0].shape))
        # np.savetxt('cnet_conv_for_in_ori.txt', inputs[0].flatten(), fmt='%f', delimiter=',')
        x = _as_mat(inputs[0])
        # np.savetxt('cnet_conv_for_x_ori.txt', x.flatten(), fmt='%f', delimiter=',')
        # x = inputs[0].flatten()
        # for i in n * c * h * w:
        #     if i % (28 * 28) is 0:
        #         print("\n"),
        #     print("%f ", x[i]),
        # with open('input_list.txt', 'a') as out_file:
        #     for i in range(2):
        #         input_line = "\ninput" + str(i)
        #         out_file.write(input_line)
        #         input = trainset[i][0].flatten()
        #         print(input)
        #         p_input = []
        #         for j in range(np.size(input)):
        #             p_input += str(input[j]) + ' '
        W = inputs[1]
        b = inputs[2]
        out_c, _, kh, kw = W.shape
        out_w = (w + 2 * self.ph - kh) / self.sx + 1
        out_h = (h + 2 * self.ph - kh) / self.sx + 1
        # print(self.ph, self.sx)
        # print(kh, kw)
        # print(out_h, out_w)
        output_size = out_h * out_w * out_c * n
        o = (c_float * output_size)()
        extra_size = c * kh * kh * out_w * out_h
        e = (c_float * extra_size)()

        l = layer()
        conv_p = conv_layer_t()
        l.batch = n
        l.input = data_t(c * h * w, cast(x.ctypes.data, POINTER(c_float)))
        # c_in = np.ctypeslib.as_ctypes(x)
        # l.input = data_t(n * c * h * w, c_in)
        l.output = data_t(out_h * out_w * out_c, o)
        l.bias = data_t(np.size(b), cast(b.ctypes.data, POINTER(c_float)))
        l.weight = data_t(np.size(W), cast(W.ctypes.data, POINTER(c_float)))
        l.extra = data_t(extra_size, e)
        conv_p.ih = h
        conv_p.iw = w
        conv_p.ic = c
        conv_p.oh = out_h
        conv_p.ow = out_w
        conv_p.oc = out_c
        conv_p.k = kh
        conv_p.s = self.sx
        conv_p.p = self.ph
        # print("ih"+ conv_p.ih + "iw"+ conv_p.iw + "ic"+ conv_p.ic + "oh"+ conv_p.oh + "ow"+ conv_p.ow + "oc"+ conv_p.oc + "k"+ conv_p.k +  "s"+ conv_p.s + "p"+ conv_p.p)
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
        forward_convolutional_layer(byref(l), byref(conv_p))
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
        # name = 'cnet_conv_for' + str(sec) + ".npy"
        # np.save(name, returnY.flatten())
        # print("conv forward output shape " + str(returnY.shape))

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
        W = inputs[1]
        out_c, _, kh, kw = W.shape
        W = W.flatten()
        nW = cupy.asnumpy(W)
        b = inputs[2]
        nb = cupy.asnumpy(b)

        out_w = (w + 2 * self.ph - kh) / self.sx + 1
        out_h = (h + 2 * self.ph - kh) / self.sx + 1
        output_size = out_h * out_w * out_c * n
        o = (c_float * output_size)()
        extra_size = c * kh * kh * out_w * out_h
        e = (c_float * extra_size)()


        l = layer()
        conv_p = conv_layer_t()
        l.batch = n
        l.input = data_t(c * h * w, cast(nx.ctypes.data, POINTER(c_float)))
        # print(n*c*h*w)
        l.output = data_t(out_h * out_w * out_c, o)
        l.bias = data_t(np.size(nb), cast(nb.ctypes.data, POINTER(c_float)))
        l.weight = data_t(np.size(nW), cast(nW.ctypes.data, POINTER(c_float)))
        l.extra = data_t(extra_size, e)
        conv_p.ih = h
        conv_p.iw = w
        conv_p.ic = c
        conv_p.oh = out_h
        conv_p.ow = out_w
        conv_p.oc = out_c
        conv_p.k = kh
        conv_p.s = self.sx
        conv_p.p = self.ph
        # string = "ih" + str(conv_p.ih) + "iw" + str(conv_p.iw) + "ic" + str(conv_p.ic) + "oh" + str(conv_p.oh) + "ow" + str(conv_p.ow) + "oc" + str(conv_p.oc) + "k" + str(conv_p.k) + "s" + str(conv_p.s) + "p" + str(conv_p.p)
        # print(string)
        # print("forward conv layer")
        forward_convolutional_layer(byref(l), byref(conv_p))
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
        # print("back conv")
        x = _as_mat(inputs[0])
        W = inputs[1].flatten()
        b = inputs[2]
        n, c, h, w = inputs[0].shape
        out_c, _, kh, kw = inputs[1].shape
        out_w = (w + 2 * self.ph - kh) / self.sx + 1
        out_h = (h + 2 * self.ph - kh) / self.sx + 1
        T_batch = n
        bias_grad = (c_float * np.size(b))()
        weight_grad = (c_float * np.size(W))()
        input_grad = (c_float * np.size(x))()
        extra_size = c * kh * kh * out_w * out_h
        e_grad = (c_float * extra_size)()
        e = (c_float * extra_size)()
        # print(grad_outputs[0].shape)

        l = layer()
        conv_p = conv_layer_t()
        l.batch = T_batch
        l.input = data_t(c * h * w, cast(x.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(out_h * out_w * out_c, grad=cast(grad_outputs[0].ctypes.data, POINTER(c_float)))
        l.bias = data_t(np.size(b), cast(b.ctypes.data, POINTER(c_float)), bias_grad)
        l.weight = data_t(np.size(W), cast(W.ctypes.data, POINTER(c_float)), weight_grad)
        l.extra = data_t(extra_size, e, e_grad)
        conv_p.ih = h
        conv_p.iw = w
        conv_p.ic = c
        conv_p.oh = out_h
        conv_p.ow = out_w
        conv_p.oc = out_c
        conv_p.k = kh
        conv_p.s = self.sx
        conv_p.p = self.ph

        # print("backward conv layer (c)")
        backward_convolutional_layer(byref(l), byref(conv_p))
        # print("end conv layer")
        gx = np.ctypeslib.as_array((c_float * np.size(x)).from_address(addressof(input_grad)))
        gW = np.ctypeslib.as_array((c_float * np.size(W)).from_address(addressof(weight_grad)))
        gb = np.ctypeslib.as_array((c_float * np.size(b)).from_address(addressof(bias_grad)))
        gx = gx.reshape(inputs[0].shape)
        gW = gW.reshape(inputs[1].shape)
        rgx = np.copy(gx)
        rgW = np.copy(gW)
        rgb = np.copy(gb)
        # time.sleep(0.1)
        # sec = time.time()
        # namex = 'cnet_conv_back_x' + str(sec) + ".npy"
        # namew = 'cnet_conv_back_w' + str(sec) + ".npy"
        # nameb = 'cnet_conv_back_b' + str(sec) + ".npy"
        # np.save(namex, rgx.flatten())
        # np.save(namew, rgW.flatten())
        # np.save(nameb, rgb.flatten())
        return rgx, rgW, rgb

    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        nx = cupy.asnumpy(x)
        W = inputs[1].flatten()
        nW = cupy.asnumpy(W)
        b = inputs[2]
        nb = cupy.asnumpy(b)
        out_g = cupy.asnumpy(grad_outputs)
        # print(grad_outputs[0].shape)
        n, c, h, w = inputs[0].shape
        out_c, _, kh, kw = inputs[1].shape
        out_w = (w + 2 * self.ph - kh) / self.sx + 1
        out_h = (h + 2 * self.ph - kh) / self.sx + 1

        T_batch = n
        bias_grad = (c_float * np.size(b))()
        weight_grad = (c_float * np.size(W))()
        input_grad = (c_float * np.size(x))()
        extra_size = c * kh * kh * out_w * out_h
        e_grad = (c_float * extra_size)()
        e = (c_float * extra_size)()

        l = layer()
        conv_p = conv_layer_t()
        l.batch = T_batch
        l.input = data_t(c * h * w, cast(nx.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(out_h * out_w * out_c, grad=cast(out_g.ctypes.data, POINTER(c_float)))
        l.bias = data_t(np.size(nb), cast(nb.ctypes.data, POINTER(c_float)), bias_grad)
        l.weight = data_t(np.size(nW), cast(nW.ctypes.data, POINTER(c_float)), weight_grad)
        l.extra = data_t(extra_size, e, e_grad)
        conv_p.ih = h
        conv_p.iw = w
        conv_p.ic = c
        conv_p.oh = out_h
        conv_p.ow = out_w
        conv_p.oc = out_c
        conv_p.k = kh
        conv_p.s = self.sx
        conv_p.p = self.ph

        # print("backward conv layer (g)")
        backward_convolutional_layer(byref(l), byref(conv_p))
        # print("end conv layer")
        gx = np.ctypeslib.as_array((c_float * np.size(x)).from_address(addressof(input_grad)))
        gW = np.ctypeslib.as_array((c_float * np.size(W)).from_address(addressof(weight_grad)))
        gb = np.ctypeslib.as_array((c_float * np.size(b)).from_address(addressof(bias_grad)))
        ygb = cupy.array(gb)
        gx = gx.reshape(inputs[0].shape)
        ygx = cupy.array(gx)
        gW2 = gW.reshape(inputs[1].shape)
        gw_return = np.copy(gW2)
        ygw_return = cupy.array(gw_return)

        return ygx, ygw_return, ygb


def cnet_convolution_2d(x, W, b=None, stride=1, pad=0, use_cudnn=True):
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
    func = CnetConvolution2DFunction(stride, pad, use_cudnn)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
    # if b is None:
    #     return DarknetConvolution2DFunction()(x, W)
    # else:
    #     return DarknetConvolution2DFunction()(x, W, b)
