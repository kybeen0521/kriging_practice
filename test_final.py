#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ordinarykriging as ok


def plot_results_compare(predicted, data) -> None:
    """
    예측값과 예측 오차를 2x2 화면으로 시각화.
    데이터 점 표시 여부 비교.

    Args:
        predicted: Kriging 예측 결과 (BasicGrid 객체)
        data: 원본 데이터 (DataFrame)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    X = sorted(set(predicted.grid.x))
    Y = sorted(set(predicted.grid.y))
    Z = np.reshape(predicted.grid.v, (len(Y), len(X)))
    Z_err = np.reshape(predicted.grid.e, (len(Y), len(X)))

    # 1. Predicted + Data
    c1 = axes[0, 0].contourf(X, Y, Z, levels=15, cmap="viridis")
    axes[0, 0].scatter(data["x"], data["y"], c="red", s=20, label="Data Points")
    axes[0, 0].legend()
    axes[0, 0].set_title("Predicted Grid + Data")
    fig.colorbar(c1, ax=axes[0, 0])

    # 2. Predicted only
    c2 = axes[0, 1].contourf(X, Y, Z, levels=15, cmap="viridis")
    axes[0, 1].set_title("Predicted Grid Only")
    fig.colorbar(c2, ax=axes[0, 1])

    # 3. Error + Data
    c3 = axes[1, 0].contourf(X, Y, Z_err, levels=15, cmap="magma")
    axes[1, 0].scatter(data["x"], data["y"], c="red", s=20, label="Data Points")
    axes[1, 0].legend()
    axes[1, 0].set_title("Prediction Error + Data")
    fig.colorbar(c3, ax=axes[1, 0])

    # 4. Error only
    c4 = axes[1, 1].contourf(X, Y, Z_err, levels=15, cmap="magma")
    axes[1, 1].set_title("Prediction Error Only")
    fig.colorbar(c4, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def run() -> None:
    start_time = time.perf_counter()

    # CSV 불러오기 및 중복 좌표 제거
    data = pd.read_csv("datas.csv").drop_duplicates(subset=["x", "y"])

    # Grid 생성
    grid = ok.Grid(data["x"].values, data["y"].values, data["v"].values)

    # 세미변량 모델 피팅
    model = grid.fitSermivariogramModel(
        modelname="Exponential",
        nlag=20,
        tnugget=1e-6,
    )

    # 예측용 정규 그리드 생성
    x, y = grid.regularBasicGrid(nx=40, ny=40)
    predicted = grid.predictedGrid(x, y, model)

    # 결과 시각화 비교
    plot_results_compare(predicted, data)

    elapsed = time.perf_counter() - start_time
    print(f"Operation performed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    run()
