# -*- coding:utf-8 -*-

N_CLIENTS = 1

net = "vgg16"
if net == "vgg11":
    BATCH_SIZE = 256
    MODEL_SAVE_ITER = 100
    DATASET_PREFIX = "/data/hishiko0/hidaka/sukiyaki2/dataset/imagenetraw/train_shuffle"
    DATA_RAW_SHAPE = (3, 256, 256)
    DATA_AUG_SHAPE = (3, 224, 224)
    DATA_AUG_CENTER = False
    DATA_AUG_SCALE = 1.0
    DATA_MEAN_FILE = "imagenet_mean.npy"
    WEIGHT_PACK_PARAM_FILE = "../netdef/vgg11_size_eightbit.json"
    INITIAL_WEIGHT = "/var/tmp/vgg11model21874.d8"
    LR = 1e-3
elif net == "vgg16":
    BATCH_SIZE = 256
    MODEL_SAVE_ITER = 100
    DATASET_PREFIX = "/data/hishiko0/hidaka/sukiyaki2/dataset/imagenetraw/train_shuffle"
    DATA_RAW_SHAPE = (3, 256, 256)
    DATA_AUG_SHAPE = (3, 224, 224)
    DATA_AUG_CENTER = False
    DATA_AUG_SCALE = 1.0
    DATA_MEAN_FILE = "imagenet_mean.npy"
    WEIGHT_PACK_PARAM_FILE = "../netdef/vgg16_size_eightbit.json"
    INITIAL_WEIGHT = "/data/unagi0/hidaka/sukiyaki2/initial_model/vgg16_initial_from_vgg11.d8"
    LR = 1e-3
elif net == "mnist":
    BATCH_SIZE = 120
    MODEL_SAVE_ITER = 100
    DATASET_PREFIX = "/home/mil/hidaka/git/sukiyaki2/mnist/mnist_train_8bit"
    DATA_RAW_SHAPE = (1, 28, 28)
    DATA_AUG_SHAPE = (1, 28, 28)
    DATA_AUG_SCALE = 0.00390625
    DATA_AUG_CENTER = True
    DATA_MEAN_FILE = None
    WEIGHT_PACK_PARAM_FILE = "../netdef/lenet_size_raw.json"
    LR = 1e-2
    #WEIGHT_PACK_PARAM_FILE = "../netdef/lenet_size_raw.json"
    #INITIAL_WEIGHT = "/var/tmp/lenet_initial.d8"

CLIENT_BATCH_SIZE = BATCH_SIZE // N_CLIENTS
