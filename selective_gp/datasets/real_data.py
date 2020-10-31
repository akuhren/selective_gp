#!/usr/bin/env python

import os
from os.path import join as pjoin, abspath
import torch

import numpy as np
import numpy.random as rand
from numpy import genfromtxt
import pandas as pd

from .dataset import Dataset


_default_folder = abspath(pjoin(__file__, "..", "..", "..", "datasets"))
DATASET_FOLDER = os.environ.get("DATASET_FOLDER", _default_folder)


class RealData(Dataset):
    def __init__(self, X, Y, task, test_size, seed, X_test=None, Y_test=None,
                 labels=None, labels_test=None):
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))

        if X_test is None and test_size:
            mask = np.zeros(len(X), dtype=bool)
            n_test = int(len(X) * test_size)
            mask[:n_test] = True
            np.random.RandomState(seed).shuffle(mask)

            X_test = X[mask]
            Y_test = Y[mask]
            X = X[~mask]
            Y = Y[~mask]

            if labels is not None:
                labels_test = labels[mask]
                labels = labels[~mask]
            else:
                labels_test = None
        else:
            X_test = torch.empty((0, *X.shape[1:]))
            Y_test = torch.empty((0, *Y.shape[1:]))

        self.X_train = torch.as_tensor(X)
        self.Y_train = torch.as_tensor(Y)
        self.X_test = torch.as_tensor(X_test)
        self.Y_test = torch.as_tensor(Y_test)
        self.labels_train = labels
        self.labels_test = labels_test

        super().__init__(task)

    @classmethod
    def uci_kin8nm(cls, test_size=0.2, seed=None):
        data = genfromtxt(pjoin(DATASET_FOLDER, 'uci_kin8nm.csv'),
                          delimiter=',', skip_header=1)
        X, Y = data[:, :8], data[:, 8:]
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def uci_boston(cls, test_size=0.2, seed=None):
        XY = pd.read_csv(pjoin(DATASET_FOLDER, 'uci_boston.csv'),
                         delim_whitespace=True).values
        X, Y = XY[:, :-1], XY[:, -1:]
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def uci_audit(cls, test_size=0.2, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_audit.csv")
        XY = pd.read_csv(path, header=0).values
        X, Y = XY[:, :-1], XY[:, -1:].astype(float)

        # Remove location ID and risk columns
        X = np.hstack((X[:, :1], X[:, 2:-1])).astype(float)

        # Set NaN value to zero (index 642, column "Money_Value")
        X[np.isnan(X)] = 0.0

        assert np.all(np.isfinite(X))

        return cls(X, Y, "binary_classification", test_size, seed)

    @classmethod
    def uci_cervical_cancer(cls, test_size=0.2, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_cervical_cancer.csv")
        df = pd.read_csv(path)

        Y = df["Dx:Cancer"].values.astype(float).reshape(-1, 1)
        X = df.loc[:, df.columns != "Dx:Cancer"].values

        # FIXME
        X[X == "?"] = np.NAN
        X = X.astype(float)

        # Replace NaN's with 0 (although should be treated as uncertain)
        X[np.isnan(X)] = 0.0

        return cls(X, Y, "binary_classification", test_size, seed)

    @classmethod
    def uci_energy(cls, test_size=0.2, heat=True, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_energy.xlsx")
        XY = pd.read_excel(path).values
        X = XY[:, :-2]
        j = -2 if heat else -1
        Y = XY[:, j].reshape(-1, 1)
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def uci_protein(cls, test_size=0.2, heat=True, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_protein.csv")
        YX = pd.read_csv(path).values
        Y, X = YX[:, :1], YX[:, 1:]
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def uci_naval(cls, test_size=0.2, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_naval.txt")
        data = np.loadtxt(path)
        X, Y = data[:, :16], data[:, 16:17]  # what to do?
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def QPCR(cls, seed=None, test_size=0):
        df = pd.read_csv(pjoin(DATASET_FOLDER, "qpcr.txt"))
        X = df.values[:, 1:].astype(float)
        labels = df.values[:, 0]
        label_set = set(labels.tolist())
        label_dict = {l: i for i, l in enumerate(label_set)}
        Y = np.zeros((len(X), 1))
        for i, l in enumerate(labels):
            Y[i] = label_dict[l]
        return cls(X, Y, "multi_classification", test_size, seed,
                   labels=labels)

    @classmethod
    def uci_concrete(cls, test_size=0.2, seed=None):
        path = pjoin(DATASET_FOLDER, "uci_concrete.xls")
        XY = pd.read_excel(path).values
        X, Y = XY[:, :-1], XY[:, -1:]
        return cls(X, Y, 'regression', test_size, seed)

    @classmethod
    def MNIST(cls, seed=None, test_size=0, digits=range(10),
              n_observations=1000):
        YX = pd.read_csv(pjoin(DATASET_FOLDER, "mnist_test.csv")).values
        Y, X = YX[:, 0].astype(int), YX[:, 1:].astype(float)
        bools = np.array([Y == i for i in digits])
        bools = np.any(bools, axis=0)

        X = X[bools]
        Y = Y[bools]

        if n_observations < len(X):
            state = rand.RandomState(seed=seed)
            idxs = state.choice(len(Y), n_observations, replace=False)
            X = X[idxs]
            Y = Y[idxs]
        X /= X.max()

        return cls(X, Y, "multi_classification", test_size, seed,
                   labels=Y)
