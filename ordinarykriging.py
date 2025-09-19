#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from base import BasicGrid


class Grid(BasicGrid):
    """Ordinary Kriging을 수행하기 위한 Grid 클래스."""

    def __init__(self, *args):
        """
        Grid 객체 초기화.

        Args:
            *args: BasicGrid 클래스 초기화 인자
        """
        super().__init__(*args)
        self.predicted: BasicGrid | None = None

    def __invGammatrix__(self, model, coords: np.ndarray | None = None) -> np.ndarray:
        """
        Gamma 행렬을 계산하고 pseudo-inverse 반환.

        Exponential 모델을 직접 계산.

        Args:
            model: Variogram 모델 객체 (sill, range_val, nugget 포함)
            coords: 좌표 배열 (미사용, 향후 확장 가능)

        Returns:
            np.ndarray: Gamma 행렬의 pseudo-inverse
        """
        n = len(self.grid.x)
        gamma_matrix = np.zeros((n + 1, n + 1))

        for i in range(n):
            for j in range(n):
                dx = self.grid.x[i] - self.grid.x[j]
                dy = self.grid.y[i] - self.grid.y[j]
                dist = np.hypot(dx, dy)
                gamma_matrix[i, j] = (
                    model.nugget
                    + model.sill * (1 - np.exp(-dist / model.range_val))
                )

        # Ordinary Kriging 제약 조건 추가
        gamma_matrix[n, :-1] = 1
        gamma_matrix[:-1, n] = 1
        gamma_matrix[n, n] = 0

        # pseudo-inverse 사용 (Singular matrix 방지)
        return np.linalg.pinv(gamma_matrix)

    def __gammavec__(self, model, x: float, y: float) -> np.ndarray:
        """
        특정 좌표 (x, y)에 대한 Gamma 벡터 계산.

        Args:
            model: Variogram 모델 객체
            x (float): 예측할 점의 x 좌표
            y (float): 예측할 점의 y 좌표

        Returns:
            np.ndarray: Gamma 벡터
        """
        n = len(self.grid.x)
        gamma = np.zeros(n + 1)

        for i in range(n):
            dx = x - self.grid.x[i]
            dy = y - self.grid.y[i]
            dist = np.hypot(dx, dy)
            gamma[i] = model.nugget + model.sill * (1 - np.exp(-dist / model.range_val))

        gamma[n] = 1  # Ordinary Kriging 제약 조건
        return gamma

    def predictedGrid(self, X: np.ndarray, Y: np.ndarray, model) -> BasicGrid:
        """
        Ordinary Kriging 예측 수행.

        Args:
            X (np.ndarray): 예측할 x 좌표 배열
            Y (np.ndarray): 예측할 y 좌표 배열
            model: Variogram 모델 객체

        Returns:
            BasicGrid: 예측된 Grid 객체 (값과 오차 포함)
        """
        nx = len(X)
        values = np.zeros(nx)
        errors = np.zeros(nx)

        inv_gamma = self.__invGammatrix__(model)

        for k in range(nx):
            gamma_vec = self.__gammavec__(model, X[k], Y[k])
            weights = np.dot(inv_gamma, gamma_vec)
            values[k] = np.sum(weights[:-1] * self.grid.v)
            errors[k] = np.sum(weights * gamma_vec)  # Kriging variance

        self.predicted = BasicGrid(X, Y, values, errors)
        return self.predicted
