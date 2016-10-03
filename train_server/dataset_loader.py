# -*- coding:utf-8 -*-

import sys
import os
import json
import random
import numpy as np

class DatasetLoader():
    """
    Load and make binary data for sending to client
    """
    def __init__(self, file_prefix, data_raw_shape, data_aug_shape):
        with open(file_prefix + ".json", "rb") as f:
            self.labels = json.load(f)
        self.data_f = open(file_prefix + ".bin", "rb")
        self.data_raw_shape = data_raw_shape#(3,256,256) ch,w,h
        self.data_raw_size = int(np.prod(data_raw_shape))
        self.data_aug_shape = data_aug_shape#(3, 224, 224)
    
    def __len__(self):
        return len(self.labels)

    def load(self, offset, count):
        offset = offset % len(self)
        self.data_f.seek(offset * self.data_raw_size)
        cur_count = min(count, len(self) - offset)
        raw_data = self.data_f.read(self.data_raw_size * cur_count)
        labels = self.labels[offset:offset+cur_count]
        if cur_count < count:
            # read from head
            self.data_f.seek(0)
            cur_count = count - cur_count
            raw_data += self.data_f.read(self.data_raw_size * cur_count)
            labels.extend(self.labels[0:0+cur_count])
        raw_data_ary = np.fromstring(raw_data, dtype=np.uint8)
        raw_data_ary = raw_data_ary.reshape((count, ) + self.data_raw_shape)
        aug_data_ary = np.zeros((count, ) + self.data_aug_shape, dtype=raw_data_ary.dtype)

        cropw = self.data_raw_shape[1] - self.data_aug_shape[1]
        croph = self.data_raw_shape[2] - self.data_aug_shape[2]
        for i in range(count):
            top = random.randint(0, croph - 1)
            left = random.randint(0, cropw - 1)
            #ch,w,h
            aug_data_ary[i] = raw_data_ary[i, :, left:left+self.data_aug_shape[1], top:top+self.data_aug_shape[2]]

        return aug_data_ary.tobytes(), np.array(labels, dtype=np.int32).tobytes()

