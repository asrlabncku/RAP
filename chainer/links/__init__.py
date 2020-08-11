"""Collection of :class:`~chainer.Link` implementations."""


from chainer.links.activation import maxout
from chainer.links.activation import prelu
from chainer.links.connection import bias
from chainer.links.connection import bilinear
from chainer.links.connection import convolution_2d
from chainer.links.connection import convolution_nd
from chainer.links.connection import deconvolution_2d
from chainer.links.connection import deconvolution_nd
from chainer.links.connection import dilated_convolution_2d
from chainer.links.connection import embed_id
from chainer.links.connection import gru
from chainer.links.connection import highway
from chainer.links.connection import inception
from chainer.links.connection import inceptionbn
from chainer.links.connection import linear
from chainer.links.connection import lstm
from chainer.links.connection import mlp_convolution_2d
from chainer.links.connection import n_step_lstm
from chainer.links.connection import parameter
from chainer.links.connection import peephole
from chainer.links.connection import scale
from chainer.links.loss import black_out
from chainer.links.loss import crf1d
from chainer.links.loss import hierarchical_softmax
from chainer.links.loss import negative_sampling
from chainer.links.model import classifier
from chainer.links.normalization import batch_normalization
from chainer.links.eBNN import link_binary_conv_BN_BST
from chainer.links.eBNN import link_binary_conv_pool_BN_BST
from chainer.links.eBNN import link_binary_linear_BN_softmax_layer
from chainer.links.eBNN import link_binary_linear_BN_BST
from chainer.links.eBNN import link_binary_linear_softmax_layer
from chainer.links.eBNN import link_conv_BN_BST
from chainer.links.eBNN import link_conv_pool_BN_BST
from chainer.links.cnet import link_cnet_lin
from chainer.links.cnet import link_cnet_conv
from chainer.links.cnet import link_cnet_pool

# for C implement
class CLink(object):
    def generate_c(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")

    def is_bin(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")

    def buf_mem(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")

    def temp_mem(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")


Maxout = maxout.Maxout
PReLU = prelu.PReLU

Bias = bias.Bias
Bilinear = bilinear.Bilinear
Convolution2D = convolution_2d.Convolution2D
ConvolutionND = convolution_nd.ConvolutionND
Deconvolution2D = deconvolution_2d.Deconvolution2D
DeconvolutionND = deconvolution_nd.DeconvolutionND
DilatedConvolution2D = dilated_convolution_2d.DilatedConvolution2D
EmbedID = embed_id.EmbedID
GRU = gru.GRU
StatefulGRU = gru.StatefulGRU
Highway = highway.Highway
Inception = inception.Inception
InceptionBN = inceptionbn.InceptionBN
Linear = linear.Linear
LSTM = lstm.LSTM
StatelessLSTM = lstm.StatelessLSTM
MLPConvolution2D = mlp_convolution_2d.MLPConvolution2D
NStepLSTM = n_step_lstm.NStepLSTM
Parameter = parameter.Parameter
StatefulPeepholeLSTM = peephole.StatefulPeepholeLSTM
Scale = scale.Scale

BlackOut = black_out.BlackOut
CRF1d = crf1d.CRF1d
BinaryHierarchicalSoftmax = hierarchical_softmax.BinaryHierarchicalSoftmax
NegativeSampling = negative_sampling.NegativeSampling

Classifier = classifier.Classifier

BatchNormalization = batch_normalization.BatchNormalization

BinaryConvBNBST = link_binary_conv_BN_BST.BinaryConvBNBST
BinaryConvPoolBNBST = link_binary_conv_pool_BN_BST.BinaryConvPoolBNBST
ConvBNBST = link_conv_BN_BST.ConvBNBST
ConvPoolBNBST = link_conv_pool_BN_BST.ConvPoolBNBST
BinaryLinearBNBST = link_binary_linear_BN_BST.BinaryLinearBNBST
BinaryLinearBNSoftmax = link_binary_linear_BN_softmax_layer.BinaryLinearBNSoftmax
BinaryLinearSoftmax = link_binary_linear_softmax_layer.BinaryLinearSoftmax

CnetConv = link_cnet_conv.CnetConv
CnetPool = link_cnet_pool.CnetPool
CnetLin = link_cnet_lin.CnetLin