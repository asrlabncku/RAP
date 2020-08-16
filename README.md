# RAP: A Software Framework of Developing Convolutional Neural Networks for Environmental Monitoring on Resource-constrained Devices
RAP permits the construction of the CNN designs by aggregating the existing, lightweight CNN layers which are able to fit in the limited memory (e.g., several KBs of SRAM)
on the resource-constrained devices satisfying application-specific timing constrains.
## Install steps
### Build cnet library file
* Build cnet library
```commandline
cd cnet_backend
cmake .
make all
```
* <odify dll path to the path of cnet library file in
    * chainer/functions/cnet/function_cnet_convolution_2d.py
    * chainer/functions/cnet/function_cnet_linear.py
    * chainer/functions/cnet/function_cnet_maxpool.py
### Build chainer
* Follow the steps in [README_chainer.md](README_chainer.md) to build chainer.
## Execution examples
### Training
There are examples in examples/RAP.
### Inference execution
* Copy the generated c code to c_implementation.
* Modify the include files, input and parameters in Source.c.
* Compile the file. 
```bash
cd c_implementation
gcc -o exe Source.c
```