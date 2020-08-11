from chainer import cuda
from chainer import function
from chainer.utils import type_check
from ctypes import *
import numpy as np
import cupy
import time
import os
dllpath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libcnet.so'))
lib = CDLL(dllpath, RTLD_GLOBAL)

class data_t(Structure):
    _fields_ = [("size", c_int),
                ("val", POINTER(c_float)),
                ("grad", POINTER(c_float))]

class layer(Structure):
    _fields_ = [("batch", c_int),
                ("input", data_t),
                ("output", data_t),
                ("weight", data_t),
                ("bias", data_t),
                ("extra", data_t)]


forward_connected_layer = lib.fc_layer_forward
forward_connected_layer.argtypes = [POINTER(layer)]
backward_connected_layer = lib.fc_layer_backward
backward_connected_layer.argtypes = [POINTER(layer)]


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class CnetLinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype == np.float32,
            w_type.dtype == np.float32,
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == np.float32,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward_cpu(self, inputs):
        x = _as_mat(inputs[0])
        # print(inputs[0].shape)
        # x = inputs[0]
        W = inputs[1].flatten()
        b = inputs[2]
        T_output = np.size(b)
        T_input = np.size(W) / T_output
        T_batch = np.size(x) / T_input
        output_size = T_batch * T_output
        # o = np.zeros(T_output, dtype=np.float)
        o = (c_float * output_size)()

        # l = layer(T_batch, T_input, T_output, 0, 0, cast(b.ctypes.data, POINTER(c_float)), None,
        #           cast(W.ctypes.data, POINTER(c_float)), None, None, o)
        l = layer()
        l.batch = T_batch
        l.input = data_t(T_input, cast(x.ctypes.data, POINTER(c_float)))
        # print(np.size(x))
        l.output = data_t(T_output, o)
        l.bias = data_t(np.size(b), cast(b.ctypes.data, POINTER(c_float)))
        l.weight = data_t(np.size(W), cast(W.ctypes.data, POINTER(c_float)))
        # print("forward connected layer(c)")
        forward_connected_layer(byref(l))
        # print("end connected layer")

        y = np.ctypeslib.as_array((c_float * (T_batch * T_output)).from_address(addressof(o)))

        # print(y.shape)
        y = np.reshape(y, (T_batch, T_output))
        rey = np.copy(y)
        # time.sleep(0.1)
        # sec = time.time()
        # name = 'cnet_linear_for' + str(sec) + ".npy"
        # np.save(name, rey.flatten())
        return rey,

    def forward_gpu(self, inputs):
        x = _as_mat(inputs[0])
        nx = cupy.asnumpy(x)
        # x = inputs[0]
        W = inputs[1].flatten()
        nW = cupy.asnumpy(W)
        b = inputs[2]
        nb = cupy.asnumpy(b)
        T_output = np.size(b)
        T_input = np.size(W) / T_output
        T_batch = np.size(x) / T_input
        output_size = T_batch * T_output
        # o = np.zeros(T_output, dtype=np.float)
        o = (c_float * output_size)()

        # l = layer(T_batch, T_input, T_output, 0, 0, cast(nb.ctypes.data, POINTER(c_float)), None,
        #           cast(nW.ctypes.data, POINTER(c_float)), None, None, o)
        l = layer()
        l.batch = T_batch
        l.input = data_t(T_input, cast(nx.ctypes.data, POINTER(c_float)))
        # print(np.size(x))
        l.output = data_t(T_output, o)
        l.bias = data_t(np.size(b), cast(nb.ctypes.data, POINTER(c_float)))
        l.weight = data_t(np.size(W), cast(nW.ctypes.data, POINTER(c_float)))
        # print("forward connected layer(g)")
        forward_connected_layer(byref(l))
        # print("end connected layer_g")

        y = np.ctypeslib.as_array((c_float * (T_batch * T_output)).from_address(addressof(o)))
        y = np.reshape(y, (T_batch, T_output))
        cy = cupy.array(y)
        # print(y.shape)
        return cy,

    def backward_cpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        # x = _inputs[0]
        W = inputs[1].flatten()
        b = inputs[2]
        # Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)
        T_output = np.size(b)
        T_input = np.size(W) / T_output
        T_batch = np.size(x) / T_input

        bias_grad = (c_float * T_output)()
        w = np.size(W)
        weight_grad = (c_float * w)()
        d_size = T_input * T_batch
        input_grad = (c_float * d_size)()

        l = layer()
        l.batch = T_batch
        l.input = data_t(T_input, cast(x.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(T_output, grad=cast(grad_outputs[0].ctypes.data, POINTER(c_float)))
        l.bias = data_t(np.size(b), cast(b.ctypes.data, POINTER(c_float)), bias_grad)
        l.weight = data_t(np.size(W), cast(W.ctypes.data, POINTER(c_float)), weight_grad)
        # print("backward connected layer(c)")
        backward_connected_layer(byref(l))
        # print("end backward connected layer_c")
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
        # namex = 'cnet_linear_back_x' + str(sec) + ".npy"
        # namew = 'cnet_linear_back_w' + str(sec) + ".npy"
        # nameb = 'cnet_linear_back_b' + str(sec) + ".npy"
        # np.save(namex, rgx.flatten())
        # np.save(namew, rgW.flatten())
        # np.save(nameb, rgb.flatten())
        # print("linear back size")
        # print(rgx.shape)
        return rgx, rgW, rgb

    def backward_gpu(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        nx = cupy.asnumpy(x)
        # print(nx.shape)
        # x = _inputs[0]
        W = inputs[1].flatten()
        nW = cupy.asnumpy(W)
        # print(nW.shape)
        b = inputs[2]
        nb = cupy.asnumpy(b)
        out_g = cupy.asnumpy(grad_outputs[0])
        # print(nb.shape)
        # Wb = numpy.where(W>=0, 1, -1).astype(numpy.float32, copy=False)
        T_output = np.size(b)
        T_input = np.size(W) / T_output
        T_batch = np.size(x) / T_input
        bias_grad = (c_float * T_output)()
        w = np.size(W)
        weight_grad = (c_float * w)()
        d_size = T_input * T_batch
        input_grad = (c_float * d_size)()

        l = layer()
        l.batch = T_batch
        l.input = data_t(T_input, cast(nx.ctypes.data, POINTER(c_float)), input_grad)
        l.output = data_t(T_output, grad=cast(out_g.ctypes.data, POINTER(c_float)))
        l.bias = data_t(np.size(b), cast(nb.ctypes.data, POINTER(c_float)), bias_grad)
        l.weight = data_t(np.size(W), cast(nW.ctypes.data, POINTER(c_float)), weight_grad)
        # print("backward connected layer(g)")
        backward_connected_layer(byref(l))
        # print("end backward connected layer_g")
        gx = np.ctypeslib.as_array((c_float * np.size(x)).from_address(addressof(input_grad)))
        gW = np.ctypeslib.as_array((c_float * np.size(W)).from_address(addressof(weight_grad)))
        gb = np.ctypeslib.as_array((c_float * np.size(b)).from_address(addressof(bias_grad)))
        ygb = cupy.array(gb)
        gx = gx.reshape(inputs[0].shape)
        ygx = cupy.array(gx)
        gW2 = gW.reshape(inputs[1].shape)
        gw_return = np.copy(gW2)
        ygw_return = cupy.array(gw_return)
        # print(ygx)
        # print(ygw_return)
        # print(ygb)
        return ygx, ygw_return, ygb


def cnet_linear(x, W, b=None):
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
    if b is None:
        return CnetLinearFunction()(x, W)
    else:
        return CnetLinearFunction()(x, W, b)
