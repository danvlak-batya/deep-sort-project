from abc import ABC, abstractmethod

import numpy as np


class BaseReID(ABC):
    """Abstract ReID encoder."""

    @property
    @abstractmethod
    def feature_dim(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def encode(self, image, boxes):
        """
        Parameters
        ----------
        image : ndarray
            BGR image.
        boxes : ndarray
            Nx4 array of bounding boxes (x, y, w, h).

        Returns
        -------
        ndarray
            N x feature_dim L2-normalized feature vectors.
        """
        pass

    def __call__(self, image, boxes):
        return self.encode(image, boxes)
