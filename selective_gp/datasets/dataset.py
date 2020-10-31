#!/usr/bin/env python

import torch


class Dataset(object):
    def __init__(self, task_type):
        if task_type == "binary_classification":
            assert torch.all((self.Y_train == 0) | (self.Y_train == 1))
        else:
            assert task_type in ["regression", "multi_classification"]

        self.task_type = task_type
        self.input_dims = self.X_train.shape[-1]

    @property
    def training_dict(self):
        return {"X": self.X_train, "Y": self.Y_train}

    @property
    def test_dict(self):
        return {"X": self.X_test, "Y": self.Y_test}

    @property
    def device(self):
        return self.X_train.device

    def __len__(self):
        return len(self.X_train) + len(self.X_test)

    def cuda(self):
        self.X_train = self.X_train.cuda()
        self.Y_train = self.Y_train.cuda()

        self.X_test = self.X_test.cuda()
        self.Y_test = self.Y_test.cuda()

    def cpu(self):
        self.X_train = self.X_train.cpu()
        self.Y_train = self.Y_train.cpu()

        self.X_test = self.X_test.cpu()
        self.Y_test = self.Y_test.cpu()

    def add_noise(self, noise_rate=0.1):
        device = self.Y_train.device
        if self.task_type == 'regression':
            s = noise_rate * self.Y_train.std()
            self.Y_train = self.Y_train + s * torch.randn_like(self.Y_train)
        elif self.task_type == 'binary_classification':
            bern = torch.distributions.Bernoulli(noise_rate)
            mask = bern.sample(self.Y_train.shape).to(device)
            self.Y_train = torch.where(mask == 0., self.Y_train,
                                       1 - self.Y_train)
        else:
            raise NotImplementedError(
                f"{self.task_type} is not a valid task type")
