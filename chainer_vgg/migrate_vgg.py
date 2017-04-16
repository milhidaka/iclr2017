#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vgg11 trained model to vgg16 initial model
"""

import sys
import os
import numpy as np
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from chainer import serializers
import vgg

def migrate_model(src, dst):
    tmp_path = "/tmp/migrate_vgg"
    src_weights = np.load(src)
    dst_model = vgg.VGG16()
    serializers.save_npz(tmp_path, dst_model)
    dst_weights = dict(np.load(tmp_path))
    for key, ary in src_weights.items():
        dst_weights[key] = ary
    np.savez(dst, **dst_weights)
    os.remove(tmp_path)

if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2]
    migrate_model(src, dst)
