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
* Modify the path of cnet library file (dllpath) in
    * chainer/functions/cnet/function_cnet_convolution_2d.py
    * chainer/functions/cnet/function_cnet_linear.py
    * chainer/functions/cnet/function_cnet_maxpool.py
### Install dependencies
#### CUDA 8.0
[CUDA 8.0 Toolkit](https://developer.nvidia.com/cuda-80-ga2-download-archive)

* Download runfile source

```shell=
chmod +x $(filename)
sudo ./$(filename)
```
* Ignore install Nvidia graphic driver option (n)
* Other option accept (y)

(Note : Because Chainer-1.17.0 version use cudnn v5 library , and cudnn v5 is depended on CUDA-8.0)
#### cuDNN v5.1
[cudnn download page](https://developer.nvidia.com/rdp/cudnn-download)

* Login Nvidia accunt(Join by yourself)
* Select cudnn v5.1 (for CUDA 8.0) option
* Download all the following option

![](https://i.imgur.com/u8JWtm2.png)

```shell=
==create symbolic link==
tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

==install runtime library==
sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn5-doc_5.1.10-1+cuda8.0_amd64.deb

```

(Note : Because Chainer-1.17.0 based on cudnn v5.1 library)
#### Other dependencies for chainer
* cython 0.25.0
* numpy 1.16.4
* six 1.10.0
* cupy 1.0.1
* h5py 2.7.0
* setuptools 40.4.3
### Build chainer
```
python setup.py install
```
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
