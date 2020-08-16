import os

import chainer
import chainer.serializers as S
from chainer.training import Trainer_RAP
from chainer.function import *
from chainer import Sequential
from chainer import ChainRAP
import numpy
from chainer.binary_layer_link import *


folder = "models/binary_float"

nfilters_embeded = int(32)
nlayers_embeded = int(1)

lr = numpy.float64(0.001)
nepochs = int(100)
name = str("mnist_cnet_20000pic_40e")

input_dims = 1
output_dims = 10


model = Sequential()


for i in range(nlayers_embeded):
    if i == 0:
        nfilters = input_dims
        model.add(CnetConv(nfilters, nfilters_embeded, 3, 1, 0))
        # model.add(CnetPool(2, 2, 0))
        # model.add(DarknetConv(nfilters_embeded, nfilters_embeded, 3, 1, 0))
        # model.add(CnetPool(nfilters, nfilters_embeded, 2, 2, 0))
    else:
        nfilters = nfilters_embeded
        # model.add(BinaryConvPoolBNBST(nfilters, nfilters_embeded, 3, 1, 1, 3, 1, 1))
model.add(CnetLin(None, output_dims))
# model.add(DarknetLinearBNBST(None, output_dims))

model.build()

chain = ChainRAP()
chain.add_sequence(model)
chain.setup_optimizers('adam', lr)

print("Model define Over ! ")

#################################################Training####################################################


print("Training Start !")
# in_shape = (3,32,32)
# c_code = model.generate_c(in_shape)
trainset, testset = chainer.datasets.get_mnist(ndim=3)
# print(trainset[1].shape())
# print(testset[1].shape())
trainset = trainset[:20000]
testset = testset[:4000]
# np.savetxt('cnet_input.txt', trainset[0][0].flatten(), fmt='%f', delimiter=',')
# testset = testset[:1]
# with open('input_list.txt', 'a') as out_file:
#     for i in range(2):
#         input_line = "\ninput" + str(i)
#         out_file.write(input_line)
#         input = trainset[i][0].flatten()
#         print(input)
#         p_input = []
#         for j in range(np.size(input)):
#             p_input += str(input[j]) + ' '

trainer = Trainer_RAP('{}/{}'.format(folder, name), chain, trainset, testset, batchsize=100, nepoch=nepochs, resume=True, gpu=-1)
# print("Pre Gen Code Start ! ")
#
# save_file = "c10_d_inference_10f_end_pre.h"
# in_shape = (3,32,32)
# c_code = model.generate_c(in_shape)
# save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
# if not os.path.exists(save_dir) and save_dir != '':
#     os.makedirs(save_dir)
#
# with open(save_file, 'w+') as fp:
#     print("Gen Code Writing ! ")
#     fp.write(c_code)
#
# print("Pre Gen Code Over ! ")
trainer.run()

print("Training Over !")

#trainer.load_model()
#################################################Inference ##################################################


#print("Inference  Start ! ")

#trainset, testset = chainer.datasets.get_cifar10(ndim=3)
#test_iter = chainer.iterators.SerialIterator(testset, 1,repeat=False, shuffle=False)
#chain.train = False
#chain.test = True
#chain.to_gpu(0)
#result = extensions.Evaluator(test_iter, chain, device=0)()

#print("Inference Over ! ")

#for key, value in result.iteritems() :
#    print key, value


#####################################################Generate Code############################################

print("Gen Code Start ! ")

save_file = "m10_d_inference_32fcl_20000pic_end_40e.h"
in_shape = (1,28,28)
c_code = model.generate_c(in_shape)
save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
if not os.path.exists(save_dir) and save_dir != '':
    os.makedirs(save_dir)

with open(save_file, 'w+') as fp:
    print("Gen Code Writing ! ")
    fp.write(c_code)

print("Gen Code Over ! ")


# print("Inference  Start ! ")
#
# trainset, testset = chainer.datasets.get_cifar10(ndim=3)
# test_iter = chainer.iterators.SerialIterator(testset, 1,repeat=False, shuffle=False)
# chain.train = False
# chain.test = True
# chain.to_gpu(0)
# result = extensions.Evaluator(test_iter, chain, device=0)()
#
# print("Inference Over ! ")
#
# for key, value in result.iteritems() :
#    print key, value