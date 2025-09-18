#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Lionel Roubeyrie"
__contact__ = "lionel.roubeyrie@gmail.com"

import numpy as np
from matplotlib import pyplot as plt


class Model:
    """Base class for semivariogram models."""

    def __init__(
        self,
        type: str | None = None,
        sill: float | None = None,
        range_val: float | None = None,
        nugget: float | None = None,
    ) -> None:
        self.type = type
        self.sill = float(sill) if sill is not None else 0.0
        self.range_val = float(range_val) if range_val is not None else 0.0
        self.nugget = max(0.0, float(nugget) if nugget is not None else 0.0)
        self.func = np.vectorize(self.f)
        self.variance = 0.0
        self.dist: np.ndarray = np.array([])
        self.svar: np.ndarray = np.array([])

    def f(self, h: float) -> float:
        """Function defining the semivariogram model."""
        return 0.0

    def __lt__(self, other: "Model") -> bool:
        return self.variance < other.variance

    def residual(self, params: tuple[float, float, float], dist: np.ndarray, svar: np.ndarray) -> np.ndarray:
        """Compute residuals between model and actual semivariances."""
        self.dist, self.svar = dist, svar
        self.sill, self.range_val, self.nugget = params
        self.nugget = max(0.0, min(self.nugget, self.sill))
        return self.svar - self.func(self.dist)

    def plot(self) -> None:
        """Plot the fitted semivariogram against observed values."""
        h = np.arange(0, np.max(self.dist), 0.1)
        v = self.func(h)
        fig, ax = plt.subplots()
        ax.plot(h, v, "r-")
        ax.plot(self.dist, self.svar, "bo")
        ax.legend(["Fit", "True"])
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title(f"SemiVariogram\n {self.sill:.1f}*{self.type}({self.range_val:.1f})+{self.nugget:.1f}")

    @property
    def corrected_sill(self) -> float:
        """Return the sill corrected with nugget."""
        return self.nugget + self.sill


class Spherical(Model):
    """Spherical semivariogram model."""

    __type__ = "Spherical"

    def __init__(self, sill: float = None, range_val: float = None, nugget: float = None) -> None:
        super().__init__("Spherical", sill, range_val, nugget)

    def f(self, h: float) -> float:
        if h >= self.range_val:
            return self.nugget + self.sill
        r = h / self.range_val
        return self.nugget + self.sill * (1.5 * r - 0.5 * r**3)


class Exponential(Model):
    """Exponential semivariogram model."""

    __type__ = "Exponential"

    def __init__(self, sill: float = None, range_val: float = None, nugget: float = None) -> None:
        super().__init__("Exponential", sill, range_val, nugget)

    def f(self, h: float) -> float:
        r = -3 * h / self.range_val
        return self.nugget + self.sill * (1 - np.exp(r))


class Gaussian(Model):
    """Gaussian semivariogram model."""

    __type__ = "Gaussian"

    def __init__(self, sill: float = None, range_val: float = None, nugget: float = None) -> None:
        super().__init__("Gaussian", sill, range_val, nugget)

    def f(self, h: float) -> float:
        r = -3 * (h / self.range_val) ** 2
        return self.nugget + self.sill * (1 - np.exp(r))


class Pentaspherical(Model):
    """Pentaspherical semivariogram model."""

    __type__ = "Pentaspherical"

    def __init__(self, sill: float = None, range_val: float = None, nugget: float = None) -> None:
        super().__init__("Pentaspherical", sill, range_val, nugget)

    def f(self, h: float) -> float:
        if h >= self.range_val:
            return self.nugget + self.sill
        r = h / self.range_val
        return self.nugget + self.sill * ((15.0 / 8.0) * r - (5.0 / 4.0) * r**3 + (3.0 / 8.0) * r**5)


class Nugget(Model):
    """Nugget effect semivariogram model."""

    __type__ = "Nugget"

    def __init__(self, sill: float = 0.0, range_val: float = None, nugget: float = None) -> None:
        super().__init__("Nugget", sill, range_val, nugget)

    def f(self, h: float) -> float:
        return self.nugget


def getModels(sill: float = 1.0, nugget: float = 0.0, range_val: float = 100.0) -> dict[str, Model]:
    """Return dictionary of semivariogram models with specified parameters."""
    return {
        "Spherical": Spherical(sill, range_val, nugget),
        "Exponential": Exponential(sill, range_val, nugget),
        "Gaussian": Gaussian(sill, range_val, nugget),
        "Pentaspherical": Pentaspherical(sill, range_val, nugget),
        "Nugget": Nugget(sill, range_val, nugget),
    }
