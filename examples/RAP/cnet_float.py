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
nepochs = int(1)
name = str("mnist_cnet_3l")

input_dims = 1
output_dims = 10


model = Sequential()


for i in range(nlayers_embeded):
    if i == 0:
        nfilters = input_dims
        model.add(CnetConv(nfilters, nfilters_embeded, 3, 1, 0))
        model.add(CnetPool(2, 2, 0))
    else:
        nfilters = nfilters_embeded
        model.add(CnetPool(2, 2, 0))
model.add(CnetLin(None, output_dims))

model.build()

chain = ChainRAP()
chain.add_sequence(model)
chain.setup_optimizers('adam', lr)

print("Model define Over ! ")

#################################################Training####################################################


print("Training Start !")
trainset, testset = chainer.datasets.get_mnist(ndim=3)

trainer = Trainer_RAP('{}/{}'.format(folder, name), chain, trainset, testset, batchsize=100, nepoch=nepochs, resume=True, gpu=-1)
trainer.run()

print("Training Over !")

#####################################################Generate Code############################################

print("Gen Code Start ! ")

save_file = "m10_cnet.h"
in_shape = (1,28,28)
c_code = model.generate_c(in_shape)
save_dir = os.path.join(os.path.split(save_file)[:-1])[0]
if not os.path.exists(save_dir) and save_dir != '':
    os.makedirs(save_dir)

with open(save_file, 'w+') as fp:
    print("Gen Code Writing ! ")
    fp.write(c_code)

print("Gen Code Over ! ")

#################################################Inference ##################################################

print("Inference  Start ! ")

trainset, testset = chainer.datasets.get_mnist(ndim=3)
test_iter = chainer.iterators.SerialIterator(testset, 1,repeat=False, shuffle=False)
chain.train = False
chain.test = True
chain.to_gpu(0)
result = chainer.training.extensions.Evaluator(test_iter, chain, device=0)()

print("Inference Over ! ")

for key, value in result.iteritems() :
   print key, value