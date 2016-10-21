import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_float32 = npct.ndpointer(dtype=np.float32, flags='CONTIGUOUS')
array_uint8 = npct.ndpointer(dtype=np.uint8, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("libdettmers_weight_compression", ".")

# setup the return types and argument types
libcd.compression_8bit.restype = None
libcd.compression_8bit.argtypes = [array_float32, array_uint8, c_int]
libcd.decompression_8bit.restype = None
libcd.decompression_8bit.argtypes = [array_uint8, array_float32, c_int]

def compression_8bit(in_array):
    out_array = np.empty((in_array.size+4, ), dtype=np.uint8)#4 is scale factor
    libcd.compression_8bit(in_array, out_array, in_array.size)
    return out_array

def decompression_8bit(in_array):
    out_array = np.empty((in_array.size-4, ), dtype=np.float32)
    libcd.decompression_8bit(in_array, out_array, out_array.size)
    return out_array
