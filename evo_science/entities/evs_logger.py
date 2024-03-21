import logging


class EvoScienceLogger(logging.Logger):
    def __init__(self, name="CPA Logger", level=logging.NOTSET):
        super().__init__(name, level)
        self.setLevel(logging.DEBUG)
        self.addHandler(logging.StreamHandler())
