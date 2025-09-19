#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ordinarykriging as ok


def plot_results(predicted) -> None:
    """
    예측값과 예측 오차를 2분할 화면으로 시각화.

    Args:
        predicted: Kriging 예측 결과 (BasicGrid 객체)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 첫 번째 subplot: 예측값
    try:
        X = sorted(set(predicted.grid.x))
        Y = sorted(set(predicted.grid.y))
        Z = np.reshape(predicted.grid.v, (len(Y), len(X)))
        c1 = axes[0].contourf(X, Y, Z, levels=15, cmap="viridis")
    except Exception as e:
        print(f"Contour plot failed (Predicted Grid): {e}")
        c1 = axes[0].scatter(predicted.grid.x, predicted.grid.y, c=predicted.grid.v, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("Predicted Grid")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # 두 번째 subplot: 예측 오차
    try:
        Z_err = np.reshape(predicted.grid.e, (len(Y), len(X)))
        c2 = axes[1].contourf(X, Y, Z_err, levels=15, cmap="magma")
    except Exception as e:
        print(f"Contour plot failed (Error Grid): {e}")
        c2 = axes[1].scatter(predicted.grid.x, predicted.grid.y, c=predicted.grid.e, cmap="magma")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("Prediction Error Grid")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def run() -> None:
    """
    Ordinary Kriging을 수행하고 예측값 및 오차를 시각화.
    """
    start_time = time.perf_counter()

    # CSV 불러오기 및 중복 좌표 제거
    data = pd.read_csv("datas.csv").drop_duplicates(subset=["x", "y"])

    # Grid 생성
    grid = ok.Grid(data["x"].values, data["y"].values, data["v"].values)

    # 세미변량 모델 피팅 (작은 nugget 적용)
    model = grid.fitSermivariogramModel(
        modelname="Exponential",
        nlag=20,
        tnugget=1e-6,
    )

    # 예측용 정규 그리드 생성
    x, y = grid.regularBasicGrid(nx=40, ny=40)
    predicted = grid.predictedGrid(x, y, model)

    # 결과 시각화 (2분할 화면)
    plot_results(predicted)

    elapsed = time.perf_counter() - start_time
    print(f"Operation performed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    run()
