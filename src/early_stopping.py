# -*- coding: utf-8 -*-
"""early_stopping.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HJhrCWhOzjM1kWTYAC1xQiSishDDB3bK
"""

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True