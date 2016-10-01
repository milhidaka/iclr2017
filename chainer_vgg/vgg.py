#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

initialW_std = 1e-2

class VGG11(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    insize = 224

    def __init__(self):
        super(VGG11, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (64, 3, 3, 3))),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (128, 64, 3, 3))),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (256, 128, 3, 3))),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (256, 256, 3, 3))),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 256, 3, 3))),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False


    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)


        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss


class VGG16(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    insize = 224

    def __init__(self):
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (64, 3, 3, 3))),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (64, 64, 3, 3))),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (128, 64, 3, 3))),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (128, 128, 3, 3))),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (256, 128, 3, 3))),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (256, 256, 3, 3))),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (256, 256, 3, 3))),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 256, 3, 3))),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=numpy.random.normal(0, initialW_std, (512, 512, 3, 3))),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False


    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8(h)


        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

