import ordinarykriging as ok
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


def plotit(x, y, z, title):
    plt.figure()
    try:
        X = sorted(list(set(x)))
        Y = sorted(list(set(y)))
        Z = np.reshape(z, (len(Y), len(X)))
        plt.contourf(X, Y, Z, 10)
    except Exception as e:
        print("Contour plot failed:", e)
        plt.scatter(x, y, c=z)
    plt.title(title)
    plt.colorbar()
    plt.show()


def run():
    t1 = time.time()

    # CSV 불러오기
    a = pd.read_csv('datas.csv')

    # 좌표 중복 제거
    a = a.drop_duplicates(subset=['x', 'y'])

    # Grid 생성
    g = ok.Grid(a['x'].values, a['y'].values, a['v'].values)

    # 세미변량 모델 피팅 (작은 nugget 적용)
    model = g.fitSermivariogramModel(modelname='Exponential', nlag=20, tnugget=1e-6)

    # 예측을 위한 정규 그리드 생성
    x, y = g.regularBasicGrid(nx=40, ny=40)
    pg = g.predictedGrid(x, y, model)

    # 예측값과 예측 오차 시각화
    plotit(pg.grid.x, pg.grid.y, pg.grid.v, "Predicted grid")
    plotit(pg.grid.x, pg.grid.y, pg.grid.e, "Predicted Error grid")

    t2 = time.time()
    print("Operation performed in {:.2f} seconds".format(t2 - t1))


if __name__ == '__main__':
    run()
