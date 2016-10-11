# -*- coding:utf-8 -*-

import sys
import os
import numpy as np

class WeightPack:
    def __init__(self, weight_size_data):
        self.weight_size_data = weight_size_data
    
    def pack(self, weights):
        if self.weight_size_data["format"] == "raw":
            return self.pack_raw(weights)

    def unpack(self, data):
        if self.weight_size_data["format"] == "raw":
            return self.unpack_raw(data)
    
    def pack_raw(self, weights, gradient=False):
        packed = np.zeros((self.weight_size_data["total_size"] / 4, ), dtype=np.float32)
        for param_meta in self.weight_size_data["param_sizes"]:
            layer = param_meta["layer"]
            name_in_layer = param_meta["delta_param"] if gradient else param_meta["train_param"]
            weight_name = layer + "/" + name_in_layer
            weight_flatten = weights[weight_name].ravel()
            packed[param_meta["offset"]/4:(param_meta["offset"]/4+param_meta["size"]/4)] = weight_flatten
        return packed.tobytes()
    
    def unpack_raw(self, data, gradient=False):
        packed = np.fromstring(data, dtype=np.float32)
        weight_dict = {}
        for param_meta in self.weight_size_data["param_sizes"]:
            layer = param_meta["layer"]
            name_in_layer = param_meta["delta_param"] if gradient else param_meta["train_param"]
            weight_name = layer + "/" + name_in_layer
            weight_flatten = packed[param_meta["offset"]/4:(param_meta["offset"]/4+param_meta["size"]/4)]
            weight_dict[weight_name] = weight_flatten
        return weight_dict
