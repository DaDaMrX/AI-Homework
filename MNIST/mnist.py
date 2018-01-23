"""mnist.py
"""

import os
import struct
import numpy as np


class MNIST:

    def __init__(self, path='.'):
        self._train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
        self._train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
        self._test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
        self._test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    def load_training(self):
        images = self._load_images(self._train_images_path)
        labels = self._load_labels(self._train_labels_path)
        return images, labels

    def load_testing(self):
        images = self._load_images(self._test_images_path)
        labels = self._load_labels(self._test_labels_path)
        return images, labels

    def _load_images(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            rows, cols = struct.unpack('>II', f.read(8))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images

    def _load_labels(self, file_path):
        with open(file_path, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels
