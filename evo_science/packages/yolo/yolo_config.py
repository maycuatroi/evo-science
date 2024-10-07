class YoloConfig:
    def __init__(self, size: str):

        accepted_sizes = ["n", "s", "m", "l", "x"]
        if size not in accepted_sizes:
            raise ValueError(f"Unsupported size: {size}")
        self.size = size

    def get_scaling_factors(self):
        if self.size == "n":
            return 1 / 3, 1 / 4, 2.0
        elif self.size == "s":
            return 1 / 3, 1 / 2, 2.0
        elif self.size == "m":
            return 2 / 3, 3 / 4, 1.5
        elif self.size == "l":
            return 1.0, 1.0, 1.0
        elif self.size == "x":
            return 1.0, 1.25, 1.0
        else:
            raise ValueError(f"Unsupported size: {self.size}")
