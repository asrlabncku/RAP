import copy, json, types
import chainer
from chainer import layer_link as link
from chainer import layer_function as function
from chainer import binary_layer_link as binary_link
from chainer import binary_layer_function as binary_function
import numpy as np
from chainer import cuda
import inspect
from chainer.variable import Variable

class GenCodeError(BaseException):
    print "gen code failed because of putting float point layer after binary layer"

class Sequential(object):
    def __init__(self, stages=[0], weight_initializer="Normal", weight_init_std=1):
        self._layers = []
        self._stages = stages
        self.links = []

        self.weight_initializer = weight_initializer    # Normal / GlorotNormal / HeNormal
        self.weight_init_std = weight_init_std
        self.current_stage = 0

    def add(self, layer):
        if isinstance(layer, Sequential):
            self._layers.append(layer)
        elif isinstance(layer, link.Link) or isinstance(layer, function.Function):
            self._layers.append(layer)
        elif isinstance(layer, function.Activation):
            self._layers.append(layer.to_function())
        elif isinstance(layer, binary_link.BinaryLink) or isinstance(layer, binary_function.BinaryFunction):
            self._layers.append(layer)
        elif isinstance(layer, binary_function.BinaryActivation):
            self._layers.append(layer.to_function())
        else:
            raise Exception()

    def layer_from_dict(self, dict):
        if "_link" in dict:
            if hasattr(link, dict["_link"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(link, dict["_link"])(**args)
            elif hasattr(binary_link, dict["_link"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(binary_link, dict["_link"])(**args)
        if "_function" in dict:
            if hasattr(function, dict["_function"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(function, dict["_function"])(**args)
            elif hasattr(binary_function, dict["_function"]):
                args = self.dict_to_layer_init_args(dict)
                return getattr(binary_function, dict["_function"])(**args)
        raise Exception()

    def dict_to_layer_init_args(self, dict):
        args = copy.deepcopy(dict)
        remove_keys = []
        for key, value in args.iteritems():
            if key[0] == "_":
                remove_keys.append(key)
        for key in remove_keys:
            del args[key]
        return args

    def get_weight_initializer(self):
        if self.weight_initializer.lower() == "normal":
            return chainer.initializers.Normal(self.weight_init_std)
        if self.weight_initializer.lower() == "glorotnormal":
            return chainer.initializers.GlorotNormal(self.weight_init_std)
        if self.weight_initializer.lower() == "henormal":
            return chainer.initializers.HeNormal(self.weight_init_std)
        raise Exception()

    def layer_to_chainer_link(self, layer):
        if hasattr(layer, "_link"):
            if layer.has_multiple_weights() == True:
                if isinstance(layer, link.GRU):
                    layer._init = self.get_weight_initializer()
                    layer._inner_init = self.get_weight_initializer()
                elif isinstance(layer, link.LSTM):
                    layer._lateral_init  = self.get_weight_initializer()
                    layer._upward_init  = self.get_weight_initializer()
                    layer._bias_init = self.get_weight_initializer()
                    layer._forget_bias_init = self.get_weight_initializer()
                elif isinstance(layer, link.StatelessLSTM):
                    layer._lateral_init  = self.get_weight_initializer()
                    layer._upward_init  = self.get_weight_initializer()
                elif isinstance(layer, link.StatefulGRU):
                    layer._init = self.get_weight_initializer()
                    layer._inner_init = self.get_weight_initializer()
            else:
                layer._initialW = self.get_weight_initializer()
            return layer.to_link()
        if hasattr(layer, "_function"):
            return layer
        raise Exception()

    def build(self):
        json = self.to_json()
        self.from_json(json)

    def to_dict(self):
        layers = []
        for layer in self._layers:
            config = layer.to_dict()
            if config.get("layers") is not None:
                layers.append(config)
                continue
            dic = {}
            for key, value in config.iteritems():
                if isinstance(value, (int, float, str, bool, type(None), tuple, list, dict)):
                    dic[key] = value
            layers.append(dic)
        return {
            "layers": layers,
            "stages": self._stages,
            "weight_initializer": self.weight_initializer,
            "weight_init_std": self.weight_init_std
        }

    def to_json(self):
        result = self.to_dict()
        return json.dumps(result, sort_keys=True, indent=4, separators=(',', ': '))

    def from_json(self, str):
        self.links = []
        self._layers = []
        self._stages = []
        attributes = {}
        dict_array = json.loads(str)
        self.from_dict(dict_array)

    def from_dict(self, dict):
        self.weight_initializer = dict["weight_initializer"]
        self.weight_init_std = dict["weight_init_std"]
        self._stages = dict["stages"]
        for i, layer_dict in enumerate(dict["layers"]):
            if layer_dict.get('layers') is not None:
                layer = Sequential(stages=layer_dict["stages"])
                layer.from_dict(layer_dict)
                self.links.append(layer)
                self._layers.append(layer)
            else:
                layer = self.layer_from_dict(layer_dict)
                link = self.layer_to_chainer_link(layer)
                self.links.append(link)
                self._layers.append(layer)

    def set_current_stage(self, stage):
        self.current_stage = stage
        for i, link in enumerate(self.links):
            if isinstance(link, Sequential):
                link.set_current_stage(stage)

    def get_current_stage(self):
        return self.current_stage

    def __call__(self, x, test=False, output_inter=False):
        bs = []
        numlinks = len(self.links)

        if output_inter:
            interm_results = [x]

        for i, link in enumerate(self.links):
            if isinstance(link, Sequential):
                # detach if in different stages
                #if reduce(lambda x,y: x and y, [stage not in link._stages for stage in self._stages]):
                if self.current_stage not in link._stages:
                    y = Variable(x.data, x.volatile)
                else:
                    y = x
                b = link(y, test=test)
                bs.append(b[0])
                # Currently not support branch inside a branch
            # elif isinstance(link, function.dropout):
            #     x = link(x, train=not test)
            elif isinstance(link, chainer.links.BatchNormalization):
                x = link(x, test=test)
            elif hasattr(link,'__call__') and 'train' in inspect.getargspec(link.__call__)[0]:
                #print("train",link)
                x = link(x, train=not test)
            elif hasattr(link,'__call__') and 'test' in inspect.getargspec(link.__call__)[0]:
                #print("test",link)
                x = link(x, test=test)
            else:
                x = link(x)
            # do not update this branch if not the current stage
            if self.current_stage not in self._stages:
                x.unchain_backward()

            if output_inter:
                interm_results.append(x.data)

        bs.append(x)

        if output_inter:
            return tuple(bs), interm_results
        else:
            return tuple(bs)

    def generate_call(self):
        link_idx = 0
        text = ""
        l = self.links[0]
        lastlink = self.links[-1]
        if lastlink.is_bin() is True:
            text += "void compute(float *input, uint8_t *output){\n"
        else:
            text += "void compute(float *input, float *output){\n"

        if l.is_bin() is True:
            text += "  {name}(input, temp1);\n".format(name=l.cname + str(link_idx))
        else:
            text += "  {name}(input, ftemp1, buf);\n".format(name=l.cname + str(link_idx))

        link_idx += 1

        pre_l = l
        for l in self.links[1:-1]:
            if pre_l.is_bin() is True and l.is_bin() is False:
                raise GenCodeError
            if link_idx % 2 == 1:
                if pre_l.is_bin() is True:
                    text += "  {name}(temp1, temp2);\n".format(name=l.cname + str(link_idx))
                elif pre_l.is_bin() is False and l.is_bin() is False:
                    text += "  {name}(ftemp1, ftemp2, buf);\n".format(name=l.cname + str(link_idx))
                else:
                    text += "  {name}(ftemp1, temp2);\n".format(name=l.cname + str(link_idx))
            else:
                if pre_l.is_bin() is True:
                    text += "  {name}(temp2, temp1);\n".format(name=l.cname + str(link_idx))
                elif pre_l.is_bin() is False and l.is_bin() is False:
                    text += "  {name}(ftemp2, ftemp1, buf);\n".format(name=l.cname + str(link_idx))
                else:
                    text += "  {name}(ftemp2, temp1);\n".format(name=l.cname + str(link_idx))
            link_idx = link_idx + 1
            pre_l = l

        l = lastlink

        if link_idx % 2 == 1:
            if l.is_bin() is True:
                text += "  {name}(temp1, output);\n".format(name=l.cname + str(link_idx))
            else:
                text += "  {name}(ftemp1, output, buf);\n".format(name=l.cname + str(link_idx))
        else:
            if l.is_bin() is True:
                text += "  {name}(temp2, output);\n".format(name=l.cname + str(link_idx))
            else:
                text += "  {name}(ftemp2, output, buf);\n".format(name=l.cname + str(link_idx))
        text += "}"

        return text

    def generate_c(self, shape, name="main", **kwargs):
        if kwargs.get("inp"):
            inp = ",".join([ p for p in inp.get("inp") ])
        else:
            inp = "0"
        text = """"""
        cnet = 0
        ebnn = 0
        for i, link in enumerate(self.links):
            if hasattr(link, 'generate_c'):
                if link.is_bin() is True:
                    ebnn = 1
                else:
                    cnet = 1
        if cnet == 1:
            text += """
#include "cnet.h"
"""
        if ebnn == 1:
            text += """
#include "ebnn.h"
"""
        h = np.random.random([1]+list(shape)).astype(np.float32)

        input_size = h.size
        binter_sizes = []
        inter_sizes = []
        buf_sizes = []
        inter_size = 0
        binter_size = 0
        buf_size = 0
        for i, link in enumerate(self.links):
            if hasattr(link, 'generate_c'):
                # float buffer
                if link.is_bin() is True:
                    binter_sizes.append(link.temp_mem(h.shape))
                else:
                    inter_sizes.append(link.temp_mem(h.shape))
                buf_sizes.append(link.buf_mem(h.shape))
                text += link.generate_c(i, h.shape)
            h = link(h, test=True)
        if inter_sizes:
            inter_size = int(np.max(inter_sizes))
        if binter_sizes:
            binter_size = int(np.max(binter_sizes))
        if buf_sizes:
            buf_size = int(np.max(buf_sizes))
        if binter_size != 0:
            text += """
uint8_t temp1[{binter_size}] = {{0}};
uint8_t temp2[{binter_size}] = {{0}};
""".format(name=name, input_size=input_size, binter_size=binter_size, inp=inp)
        if inter_size != 0:
            text += """
float ftemp1[{inter_size}] = {{0}};
float ftemp2[{inter_size}] = {{0}};
""".format(name=name, input_size=input_size, inter_size=inter_size, inp=inp)
        if buf_size != 0:
            text += """
float buf[{buf_size}] = {{0}};
""".format(buf_size=buf_size)

        text += self.generate_call()
        return text
