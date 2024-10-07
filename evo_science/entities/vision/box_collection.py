import numpy as np


class BoxCollection:
    def __init__(self, boxes: np.ndarray):
        """
        Initialize the BoxCollection with a set of boxes.

        Args:
            boxes: list of boxes
        """
        self.boxes = np.array(boxes)

    def wh2xy(self, width=640, height=640, pad_width=0, pad_height=0):
        """
        Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2) format.
        """
        xy_boxes = np.copy(self.boxes)
        xy_boxes[:, [0, 2]] = self._scale_and_shift(
            self.boxes[:, [0, 2]], self.boxes[:, 2:3], width, pad_width, subtract=True
        )
        xy_boxes[:, [1, 3]] = self._scale_and_shift(
            self.boxes[:, [1, 3]], self.boxes[:, 3:4], height, pad_height, subtract=False
        )
        return xy_boxes

    def xy2wh(self, width, height):
        """
        Convert boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height) format.
        """
        boxes = self._clip_boxes(self.boxes, width, height)
        wh_boxes = np.copy(boxes)
        wh_boxes[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        wh_boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        return self._normalize_boxes(wh_boxes, width, height)

    def candidates(self, other_boxes, min_size=2, min_area_ratio=0.1, max_aspect_ratio=100):
        """
        Determine if other_boxes are valid candidates relative to the stored boxes.

        Args:
            other_boxes: numpy array of shape (m, 4) where m is the number of other boxes

        Returns:
            boolean array of shape (n, m) where n is the number of stored boxes
        """
        w1, h1 = self.boxes[:, 2] - self.boxes[:, 0], self.boxes[:, 3] - self.boxes[:, 1]
        w2, h2 = other_boxes[:, 2] - other_boxes[:, 0], other_boxes[:, 3] - other_boxes[:, 1]

        aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        area_ratio = (w2 * h2)[:, np.newaxis] / ((w1 * h1) + 1e-16)

        return (
            (w2 > min_size)
            & (h2 > min_size)
            & (area_ratio > min_area_ratio)
            & (aspect_ratio[:, np.newaxis] < max_aspect_ratio)
        )

    def _scale_and_shift(self, values, scales, dimension, padding, subtract):
        """Helper method for scaling and shifting box coordinates."""
        operation = np.subtract if subtract else np.add
        return dimension * operation(values, scales / 2) + padding

    def _clip_boxes(self, boxes, width, height):
        """Clip box coordinates to image dimensions."""
        return np.clip(boxes, 0, [width - 1e-3, height - 1e-3, width - 1e-3, height - 1e-3])

    def _normalize_boxes(self, boxes, width, height):
        """Normalize box coordinates by image dimensions."""
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        return boxes
