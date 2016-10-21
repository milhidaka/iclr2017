# Optimizer with 8bit-compression/decompression to check learning curve change
# Received gradient to 8bit and send weight as 8bit
# based on momentum_sgd.py in Chainer

import os
import sys
import numpy as np
from chainer import cuda
from chainer.optimizers import MomentumSGD
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../train_server")
import dettmers_weight_compression

def compress_decompress(ary):
    code = dettmers_weight_compression.compression_8bit(ary)
    restored = dettmers_weight_compression.decompression_8bit(code)
    restored = restored.reshape(ary.shape)
    return restored

class MomentumSGDCompress(MomentumSGD):
    def __init__(self, *args, **kwargs):
        super(MomentumSGDCompress, self).__init__(*args, **kwargs)

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)
            state['t'] = xp.array(param.data)#true weight

    def update_one_cpu(self, param, state):
        v = state['v']
        t = state['t']
        v *= self.momentum
        comp_grad = compress_decompress(param.grad)
        v -= self.lr * comp_grad
        t += v
        param.data[:] = compress_decompress(t)

    def update_one_gpu(self, param, state):
        grad_cpu = param.grad.get()
        comp_grad = compress_decompress(grad_cpu)
        param.grad[:] = cuda.cupy.array(comp_grad)
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(param.grad, self.lr, self.momentum,
                            state['t'], state['v'])
        comp_data = compress_decompress(state['t'].get())
        param.data[:] = cuda.cupy.array(comp_data)
