# Kriging-based Geostatistical Analysis practice


## 📌 Overview
This repository provides a Python pipeline for geostatistical analysis using Ordinary Kriging.
It covers everything from grid setup, semivariogram modeling, ordinary kriging interpolation, to visualization.

The workflow consists of four main modules:

- base.py → Grid setup and semivariogram data preparation

models.py → Semivariogram model definitions (Spherical, Exponential, Gaussian, etc.)

ordinarykriging.py → Implementation of the Ordinary Kriging method

test.py → End-to-end execution script (data loading, fitting, prediction, visualization)

---

## ✨ Features

- Random subsampling of spatial data

- Computation of pairwise distances, angles, and semivariances

- Multiple semivariogram models:

- Spherical

- Exponential

- Gaussian

- Pentaspherical

- Nugget

- Least-squares fitting of variogram models

- Ordinary Kriging interpolation with pseudo-inverse solver (robust to singularities)

- CSV export of grid data

- Visualization of predicted surfaces and error maps

---

## 🔄 Workflow

Input data (datas.csv, with columns x, y, v)
↓ Step 1: Grid Construction → base.py
↓ Step 2: Variogram Modeling → models.py
↓ Step 3: Ordinary Kriging Interpolation → ordinarykriging.py
↓ Step 4: Execution & Visualization → test.py
↓ Output: Predicted values, error maps, logs

---

## 📝 Step Descriptions

### 1️⃣ base.py – Grid Construction

Reads (x, y, v) input data

Computes pairwise distances, angles, and semivariances

Prepares structured grid for variogram and kriging

### 2️⃣ models.py – Variogram Models

- Defines theoretical models:

- Spherical

- Exponential

- Gaussian

- Pentaspherical

- Nugget

- Provides functions for curve fitting

### 3️⃣ ordinarykriging.py – Kriging

- Constructs kriging system matrix

- Solves with pseudo-inverse to handle singularities

- Outputs predictions and estimation variances

### 4️⃣ test.py – Run & Visualize

- Loads data from datas.csv

- Fits semivariogram model

- Runs kriging interpolation

- Plots predicted surface and error map


---
## 📂 Project Structure
```
├── base.py              # Grid setup & semivariogram data preparation
├── models.py            # Semivariogram models
├── ordinarykriging.py   # Ordinary Kriging implementation
├── test.py              # End-to-end pipeline script
├── datas.csv            # Example input data
└── README.md            # Documentation
```
---


## 👤 Author
**Yongbeen Kim (김용빈)**  
Researcher, Intelligent Mechatronics Research Center, KETI


📅 Document last updated 2025.09.20

