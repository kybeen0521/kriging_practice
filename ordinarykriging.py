#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from base import BasicGrid

class Grid(BasicGrid):
    def __init__(self, *args):
        super(Grid, self).__init__(*args)
        self.predicted = None

    def __invGammatrix__(self, model, coords=None):
        """
        Gamma 행렬 계산 후 pseudo-inverse 반환
        Exponential 모델을 직접 계산
        """
        n = len(self.grid.x)
        m = np.zeros((n+1, n+1))

        for i in range(n):
            for j in range(n):
                dx = self.grid.x[i] - self.grid.x[j]
                dy = self.grid.y[i] - self.grid.y[j]
                dist = np.sqrt(dx**2 + dy**2)
                # Exponential 세미변량 계산 직접 수행
                m[i, j] = model.nugget + model.sill * (1 - np.exp(-dist / model.range_val))

        # Ordinary Kriging 제약 추가
        m[n, :-1] = 1
        m[:-1, n] = 1
        m[n, n] = 0

        # Singular matrix 방지
        invG = np.linalg.pinv(m)
        return invG

    def __gammavec__(self, model, x, y):
        """
        특정 좌표(x, y)에 대한 Gamma 벡터 계산
        """
        n = len(self.grid.x)
        gamma = np.zeros(n+1)

        for i in range(n):
            dx = x - self.grid.x[i]
            dy = y - self.grid.y[i]
            dist = np.sqrt(dx**2 + dy**2)
            gamma[i] = model.nugget + model.sill * (1 - np.exp(-dist / model.range_val))

        gamma[n] = 1  # Ordinary Kriging 제약
        return gamma

    def predictedGrid(self, X, Y, model):
        """
        Ordinary Kriging 예측 수행
        """
        nx = len(X)
        values = np.zeros(nx)
        errors = np.zeros(nx)

        # Gamma 행렬 역행렬
        invG = self.__invGammatrix__(model)

        for k in range(nx):
            gamma = self.__gammavec__(model, X[k], Y[k])
            weights = np.dot(invG, gamma)
            values[k] = np.sum(weights[:-1] * self.grid.v)
            errors[k] = np.sum(weights * gamma)  # Kriging variance

        # 결과를 새로운 Grid 객체에 저장
        self.predicted = BasicGrid(X, Y, values, errors)
        return self.predicted
