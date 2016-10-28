#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import argparse
import single_train_loop

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initmodel", required=True)
    parser.add_argument("--savetmpl", required=True)
    parser.add_argument("--initt", type=int, default=0)
    args = parser.parse_args()
    # launch var server
    # launch static content server
    # start training loop
    single_train_loop.train_loop(initial_weight_path=args.initmodel, weight_save_path_tmpl=args.savetmpl, initial_t=args.initt)

if __name__ == '__main__':
    run()
