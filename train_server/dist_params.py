# -*- coding:utf-8 -*-

N_CLIENTS = 2
BATCH_SIZE = 32
CLIENT_BATCH_SIZE = BATCH_SIZE // N_CLIENTS
DATASET_PREFIX = "/data/hishiko0/hidaka/sukiyaki2/dataset/imagenetraw/train_shuffle"
DATA_RAW_SHAPE = (3, 256, 256)
DATA_AUG_SHAPE = (3, 224, 224)
