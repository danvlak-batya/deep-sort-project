from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Abstract person detector returning (tlwh, confidence) per detection."""

    PERSON_CLASS_ID = 0

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def detect(self, image):
        """
        Parameters
        ----------
        image : ndarray
            BGR image (H, W, 3).

        Returns
        -------
        list of tuple
            Each element is (tlwh, confidence) with tlwh = (x, y, w, h).
        """
        pass

    def detect_batch(self, images):
        return [self.detect(img) for img in images]
