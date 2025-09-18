#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Lionel Roubeyrie"
__contact__ = "lionel.roubeyrie at gmail dot com"

import numpy as np
from matplotlib import pyplot as plt


class Model(object):
    def __init__(self, type=None, sill=None, range_val=None, nugget=None):
        self.type = type
        self.sill = float(sill)
        self.range_val = float(range_val)
        self.nugget = float(nugget)
        if self.nugget < 0:
            self.nugget = 0
        self.func = np.vectorize(self.f)
        self.variance = 0

    def f(self, h):
        return 0

    # Python 3에서는 __cmp__ 제거
    def __lt__(self, other):
        return self.variance < other.variance

    def residual(self, params, dist, svar):
        self.dist = dist
        self.svar = svar
        self.sill, self.range_val, self.nugget = params
        if self.nugget < 0:
            self.nugget = 0
        if self.nugget > self.sill:
            self.nugget = self.sill
        err = self.svar - self.func(self.dist)
        return err

    def plot(self):
        h = np.arange(0, np.max(self.dist), 0.1)
        v = self.func(h)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h, v, 'r-')
        ax.plot(self.dist, self.svar, 'bo')
        plt.legend(['Fit', 'True'])
        plt.xlabel("distance")
        plt.ylabel("semivariance")
        plt.title("SemiVariogram\n %.1f*%s(%.1f)+%.1f" % (self.sill, self.type,
                                                           self.range_val, self.nugget))

    def getCorrectedSill(self):
        return self.nugget + self.sill

    corrected_sill = property(getCorrectedSill)


class Spherical(Model):
    __type__ = "Spherical"
    def __init__(self, sill=None, range_val=None, nugget=None):
        super(Spherical, self).__init__("Spherical", sill, range_val, nugget)

    def f(self, h):
        if self.range_val <= h:
            return self.nugget + self.sill
        else:
            r = float(h)/self.range_val
            return self.nugget + self.sill * ((1.5*r) - (0.5*(r**3)))


class Exponential(Model):
    __type__ = "Exponential"
    def __init__(self, sill=None, range_val=None, nugget=None):
        super(Exponential, self).__init__("Exponential", sill, range_val, nugget)

    def f(self, h):
        r = -(3*h)/self.range_val
        try:
            return self.nugget + self.sill*(1 - np.exp(r))
        except OverflowError:
            return -np.infty


class Gaussian(Model):
    __type__ = "Gaussian"
    def __init__(self, sill=None, range_val=None, nugget=None):
        super(Gaussian, self).__init__("Gaussian", sill, range_val, nugget)

    def f(self, h):
        r = -3*((h/self.range_val)**2)
        return self.nugget + self.sill*(1 - np.exp(r))


class Pentaspherical(Model):
    __type__ = "Pentaspherical"
    def __init__(self, sill=None, range_val=None, nugget=None):
        super(Pentaspherical, self).__init__("Pentaspherical", sill, range_val, nugget)

    def f(self, h):
        if self.range_val <= h:
            return self.nugget + self.sill
        else:
            r = float(h)/self.range_val
            return self.nugget + self.sill*((15.0/8.0)*r - (5.0/4.0)*r**3 + (3.0/8.0)*r**5)


class Nugget(Model):
    __type__ = "Nugget"
    def __init__(self, sill=0, range_val=None, nugget=None):
        super(Nugget, self).__init__("Nugget", sill, range_val, nugget)

    def f(self, h):
        return self.nugget


def getModels(sill=1.0, nugget=0.0, range_val=100.0):
    models = {
        "Spherical": Spherical(sill, range_val, nugget),
        "Exponential": Exponential(sill, range_val, nugget),
        "Gaussian": Gaussian(sill, range_val, nugget),
        "Pentaspherical": Pentaspherical(sill, range_val, nugget),
        "Nugget": Nugget(sill, range_val, nugget)
    }
    return models
