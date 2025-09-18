#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Lionel Roubeyrie"
__contact__ = "lionel.roubeyrie at gmail dot com"

import models
import numpy as np
import scipy.spatial as spatial
import scipy.optimize as optimize


def randin(grid, pcent):
    """Returns pcent values of a grid where indices are randomly
    selected only once.
    """
    n = grid.shape[0]-1
    if n == 0:
        return None
    ind = list()
    m = int(n*pcent/100)
    while len(ind) != m:
        a = np.random.randint(0, n+1)
        if a not in ind:
            ind.append(a)
    res = grid[ind].ravel()
    return res


def pprint(text):
    print("--------------------------")
    for l, t in text:
        print(" {} : {}".format(l, t))
    print("--------------------------\n")


class BasicGrid(object):
    def __init__(self, *args):
        if len(args) == 1:  # U
            self.data = np.asarray(args[0])
        elif len(args) == 3:  # X, Y, V
            self.data = np.array([args[0], args[1], args[2], np.zeros_like(args[0])]).T
        elif len(args) == 4:  # X, Y, V, E
            self.data = np.array(args).T
        else:
            raise ValueError("Bad input datas formats")
        names = ("x", "y", "v", "e")
        self.grid = np.rec.fromrecords(self.data, names=names)
        pprint([
            ["Number of point", self.grid.x.shape[0]],
            ["Standard deviation of the grid", np.std(self.grid.v)]
        ])
        self.infos = self.compInfos(self.grid)

    def distance(self, a, b):
        return np.sqrt(np.square(a.x - b.x) + np.square(a.y - b.y))

    def semiVariance(self, a, b):
        return 0.5 * np.square(a.v - b.v)

    def angle(self, a, b):
        # 배열/스칼라 모두 처리 가능
        dx = b.x - a.x
        dy = b.y - a.y
        angle = np.degrees(np.arctan2(dy, dx))  # dx=0도 안전하게 처리
        return angle

    def compInfos(self, grid):
        print("Analysing grid. Please wait...")
        n = grid.shape[0]
        id_i = np.array([], dtype=int)
        id_j = np.array([], dtype=int)
        dist = np.array([], dtype=float)
        svar = np.array([], dtype=float)
        ang = np.array([], dtype=float)
        for k in range(n-1):
            j = range(k+1, n)
            i = np.repeat(k, len(j))
            pi = grid[i]
            pj = grid[j]
            id_i = np.append(id_i, i)
            id_j = np.append(id_j, j)
            dist = np.append(dist, self.distance(pi, pj))
            svar = np.append(svar, self.semiVariance(pi, pj))
            ang = np.append(ang, self.angle(pi, pj))
        p = np.vstack((id_i, id_j, dist, svar, ang))
        names = ("id_i", "id_j", "dist", "svar", "ang")
        p = np.rec.fromrecords(p.T, names=names)
        return p

    def fitSermivariogramModel(self, modelname, nlag=15, tsill=None,
                               trange=None, tnugget=0.0):
        dist = self.infos.dist
        svar = self.infos.svar

        if nlag is None:
            nlag = len(dist)
        if nlag < 1:
            raise ValueError("nlag must be >=1")

        if tsill is None:
            tsill = 9/10*np.max(svar)
        if trange is None:
            trange = 0.5*np.max(dist)

        sortind = np.argsort(dist)
        sortdist = dist[sortind]
        sortsvar = svar[sortind]

        index = sortdist.searchsorted(np.linspace(0, dist.max(), nlag+1))
        dist = [sortdist[index[i-1]:index[i]].mean() for i in range(1, len(index))]
        svar = [sortsvar[index[i-1]:index[i]].mean() for i in range(1, len(index))]

        # getModels 호출 시 range_val 사용
        model = models.getModels(sill=tsill, range_val=trange, nugget=tnugget)[modelname]

        # 모델 속성에 맞춰 params 정의
        params = (model.sill, model.range_val, model.nugget)

        lstsqResult = optimize.leastsq(model.residual, params, args=(dist, svar), full_output=0)

        if lstsqResult != 1:
            p = lstsqResult[0]
            # 최적화 후 모델 속성 업데이트
            model.sill, model.range_val, model.nugget = p
            if model.range_val > 0:
                squareDeviates = model.residual(p, dist, svar)**2
                denom = 1 / float(len(svar)-len(p))
                model.variance = denom*sum(squareDeviates)
                pprint([["Model Type", model.type], ["Sill", model.sill],
                        ["Range", model.range_val], ["Nugget", model.nugget]])
                return model
            else:
                pprint([["Error", "Computed range <=0 :("]])
                return None
        else:
            pprint([["Error", "Bad fitting computation :("]])
            return None

    def tofile(self, fname):
        with open(fname, "w") as f:
            f.write("X,Y,V,E\n")
            for p in self.grid:
                f.write("{},{},{},{}\n".format(p.x, p.y, p.v, p.e))

    def regularBasicGrid(self, xmin=None, ymin=None, xmax=None, ymax=None,
                         nx=30, ny=30):
        if xmin is None:
            xmin = self.grid.x.min()
        if ymin is None:
            ymin = self.grid.y.min()
        if xmax is None:
            xmax = self.grid.x.max()
        if ymax is None:
            ymax = self.grid.y.max()
        X = np.linspace(xmin, xmax, nx)
        Y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        return X, Y
