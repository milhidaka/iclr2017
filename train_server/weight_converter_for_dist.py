#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Weight file conversion for distributed training

import sys
import os
import json
import argparse
import re
import numpy as np
from weight_pack import WeightPack

def sukiyaki_unpack(binary):
    """
    load key-flatarray pairs
    """
    eoh = binary.find('\0')
    header_str = binary[:eoh]
    header = json.loads(header_str)
    pairs = {}
    for key, pos in header.items():
        pairs[key] = np.fromstring(binary[pos["offset"]:pos["offset"]+pos["size"]], dtype=np.float32)
    return pairs

def sukiyaki_pack(pairs):
    """
    key-flatarray pairs to serialized byte string
    """
    header_size = 65536
    while True:
        header = {}
        offset = header_size
        elements = []
        for key, ary in pairs.items():
            assert ary.dtype == np.float32
            ary_size = ary.size * 4
            header[key] = {"offset":offset, "size":ary_size}
            elements.append(ary.tobytes(order="C"))
            offset += ary_size
        header_str = json.dumps(header)
        if len(header_str) < header_size:
            break
        header_size *= 2
    nullstr = "\0" * (header_size - len(header_str))
    elements.insert(0, nullstr)
    elements.insert(0, header_str)
    return "".join(elements)

def s2d(src, dst, packer):
    with open(src, "rb") as f:
        weights = sukiyaki_unpack(f.read())
    
    d_packed = packer.pack(weights)
    with open(dst, "wb") as f:
        f.write(d_packed)

def d2s(src, dst, packer):
    with open(src, "rb") as f:
        weights = packer.unpack(f.read())
    
    s_packed = sukiyaki_pack(weights)
    with open(dst, "wb") as f:
        f.write(s_packed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", help="s2d: sukiyaki to distributed, d2s: distributed to sukiyaki")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("weight_pack_meta")
    args = parser.parse_args()
    packer = WeightPack(json.load(open(args.weight_pack_meta)))
    if args.direction == "s2d":
        s2d(args.src, args.dst, packer)
    elif args.direction == "d2s":
        d2s(args.src, args.dst, packer)
    else:
        raise KeyError("Unknown direction")
