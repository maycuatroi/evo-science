import numpy as np

from evo_science.dl.layers.base_layer import BaseLayer


class CNN(BaseLayer):
    def __init__(self, filters=np.random.randn(3, 3, 3), stride=1):
        self.filters = filters
        self.num_filters, self.filter_size, _ = self.filters
        self.stride = stride

        super().__init__()

    def forward(
        self,
        x,
    ):
        num_filters, filter_size, _ = self.filters.shape
        h, w = x.shape
        out_dim = (h - filter_size) // self.stride + 1
        conv_out = np.zeros((num_filters, out_dim, out_dim))
        for f in range(num_filters):
            for i in range(0, h - filter_size + 1, self.stride):
                for j in range(0, w - filter_size + 1, self.stride):
                    conv_out[f, i // self.stride, j // self.stride] = np.sum(
                        x[i : i + filter_size, j : j + filter_size] * self.filters[f]
                    )
        return conv_out
