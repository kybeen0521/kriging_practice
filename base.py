#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Lionel Roubeyrie"
__contact__ = "lionel.roubeyrie@gmail.com"

import models
import numpy as np
import scipy.optimize as optimize


def randin(grid: np.ndarray, pcent: float) -> np.ndarray | None:
    """Return pcent values of a grid with unique random indices."""
    n = grid.shape[0] - 1
    if n == 0:
        return None

    m = int(n * pcent / 100)
    ind = []
    while len(ind) != m:
        a = np.random.randint(0, n + 1)
        if a not in ind:
            ind.append(a)
    return grid[ind].ravel()


def pprint(text: list[tuple[str, float | str]]) -> None:
    """Pretty print key-value information."""
    print("-" * 26)
    for label, value in text:
        print(f" {label} : {value}")
    print("-" * 26 + "\n")


class BasicGrid:
    """Class representing a basic grid for geostatistical analysis."""


    def __init__(self, *args):
        """
        Initialize grid:
        - 1 argument: U array
        - 3 arguments: X, Y, V
        - 4 arguments: X, Y, V, E
        """
        if len(args) == 1:
            self.data = np.asarray(args[0])
        elif len(args) == 3:
            x, y, v = args
            self.data = np.array([x, y, v, np.zeros_like(x)]).T
        elif len(args) == 4:
            self.data = np.array(args).T
        else:
            raise ValueError("Bad input data format. Must be 1, 3, or 4 arrays.")

        names = ("x", "y", "v", "e")
        self.grid = np.rec.fromrecords(self.data, names=names)

        pprint([
            ("Number of points", self.grid.x.shape[0]),
            ("Standard deviation of the grid", np.std(self.grid.v))
        ])

        self.infos = self.compInfos(self.grid)


    def distance(self, a, b) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


    def semiVariance(self, a, b) -> float:
        """Compute semi-variance between two points."""
        return 0.5 * (a.v - b.v) ** 2


    def angle(self, a, b) -> float:
        """Compute angle (in degrees) between two points."""
        dx = b.x - a.x
        dy = b.y - a.y
        return np.degrees(np.arctan2(dy, dx))


    def compInfos(self, grid) -> np.recarray:
        """Compute pairwise distances, semi-variances, and angles."""
        print("Analysing grid. Please wait...")
        n = grid.shape[0]
        id_i, id_j, dist, svar, ang = [], [], [], [], []

        for k in range(n - 1):
            for j in range(k + 1, n):
                id_i.append(k)
                id_j.append(j)
                dist.append(self.distance(grid[k], grid[j]))
                svar.append(self.semiVariance(grid[k], grid[j]))
                ang.append(self.angle(grid[k], grid[j]))

        return np.rec.fromarrays([id_i, id_j, dist, svar, ang],
                                 names=("id_i", "id_j", "dist", "svar", "ang"))


    def fitSermivariogramModel(
        self,
        modelname: str,
        nlag: int = 15,
        tsill: float = None,
        trange: float = None,
        tnugget: float = 0.0
    ):
        """Fit a semivariogram model using least squares optimization."""
        dist, svar = self.infos.dist, self.infos.svar

        if nlag is None:
            nlag = len(dist)
        if nlag < 1:
            raise ValueError("nlag must be >=1")

        tsill = tsill or 0.9 * np.max(svar)
        trange = trange or 0.5 * np.max(dist)

        sortind = np.argsort(dist)
        sortdist, sortsvar = dist[sortind], svar[sortind]

        index = sortdist.searchsorted(np.linspace(0, dist.max(), nlag + 1))
        dist_bin = [sortdist[index[i - 1]:index[i]].mean() for i in range(1, len(index))]
        svar_bin = [sortsvar[index[i - 1]:index[i]].mean() for i in range(1, len(index))]

        model = models.getModels(sill=tsill, range_val=trange, nugget=tnugget)[modelname]
        params = (model.sill, model.range_val, model.nugget)

        lstsqResult = optimize.leastsq(model.residual, params, args=(dist_bin, svar_bin), full_output=0)

        if lstsqResult != 1:
            p = lstsqResult[0]
            model.sill, model.range_val, model.nugget = p
            if model.range_val > 0:
                square_deviates = model.residual(p, dist_bin, svar_bin) ** 2
                model.variance = sum(square_deviates) / (len(svar_bin) - len(p))
                pprint([
                    ("Model Type", model.type),
                    ("Sill", model.sill),
                    ("Range", model.range_val),
                    ("Nugget", model.nugget)
                ])
                return model
            else:
                pprint([("Error", "Computed range <= 0")])
                return None
        else:
            pprint([("Error", "Bad fitting computation")])
            return None


    def tofile(self, fname: str) -> None:
        """Save grid to CSV file."""
        with open(fname, "w") as f:
            f.write("X,Y,V,E\n")
            for p in self.grid:
                f.write(f"{p.x},{p.y},{p.v},{p.e}\n")


    def regularBasicGrid(
        self,
        xmin: float = None,
        ymin: float = None,
        xmax: float = None,
        ymax: float = None,
        nx: int = 30,
        ny: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a regular grid (X, Y) over specified bounds."""
        xmin = xmin if xmin is not None else self.grid.x.min()
        ymin = ymin if ymin is not None else self.grid.y.min()
        xmax = xmax if xmax is not None else self.grid.x.max()
        ymax = ymax if ymax is not None else self.grid.y.max()

        X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                           np.linspace(ymin, ymax, ny))
        return X.flatten(), Y.flatten()
