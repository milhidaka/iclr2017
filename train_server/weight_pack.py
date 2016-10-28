# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import dettmers_weight_compression

class WeightPack:
    def __init__(self, weight_size_data):
        self.weight_size_data = weight_size_data
    
    def pack(self, weights):
        if self.weight_size_data["format"] == "raw":
            return self.pack_raw(weights)
        elif self.weight_size_data["format"] == "eightbit":
            return self.pack_eightbit(weights)

    def unpack(self, data):
        if self.weight_size_data["format"] == "raw":
            return self.unpack_raw(data)
        elif self.weight_size_data["format"] == "eightbit":
            return self.unpack_eightbit(data)
    
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

    def pack_eightbit(self, weights, gradient=False):
        packed = np.zeros((self.weight_size_data["total_size"], ), dtype=np.uint8)
        for param_meta in self.weight_size_data["param_sizes"]:
            layer = param_meta["layer"]
            name_in_layer = param_meta["delta_param"] if gradient else param_meta["train_param"]
            weight_name = layer + "/" + name_in_layer
            weight_flatten = weights[weight_name].ravel()
            # size includes 4 byte magnitude store area
            #packed[param_meta["offset"]:(param_meta["offset"]+param_meta["size"]-4)] = naive_eightbit_encoder(weight_flatten)
            packed[param_meta["offset"]:(param_meta["offset"]+param_meta["size"])] = dettmers_weight_compression.compression_8bit(weight_flatten)
        return packed.tobytes()
    
    def unpack_eightbit(self, data, gradient=False):
        packed = np.fromstring(data, dtype=np.uint8)
        weight_dict = {}
        for param_meta in self.weight_size_data["param_sizes"]:
            layer = param_meta["layer"]
            name_in_layer = param_meta["delta_param"] if gradient else param_meta["train_param"]
            weight_name = layer + "/" + name_in_layer
            #weight_flatten = naive_eightbit_decoder(packed[param_meta["offset"]:(param_meta["offset"]+param_meta["size"]-4)])
            weight_flatten = dettmers_weight_compression.decompression_8bit(packed[param_meta["offset"]:(param_meta["offset"]+param_meta["size"])])
            weight_dict[weight_name] = weight_flatten
        return weight_dict

def naive_eightbit_encoder(array_float):
    return np.zeros(array_float.shape, dtype=np.uint8)
    s = np.sign(array_float).astype(np.uint8)
    flag = s & 0x80
    y = np.log2(np.fabs(array_float))
    exponent = np.clip(y + 64, 0, 127).astype(np.uint8)
    packed = flag + exponent
    return packed#uint8

def naive_eightbit_decoder(array_packed):
    return np.zeros(array_packed.shape, dtype=np.float32)
    s = (array_packed >> 7) * np.float32(-2.) + 1.
    y = np.exp2((array_packed & 0x7F).astype(np.float32) - 64.)
    array_float = s * y
    return array_float
