import torch


class Profiler:
    def __init__(self, model, input_size: int, batch_size: int, num_classes: int):
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def profile(self):
        self.model.eval()
        self.model(torch.zeros((self.batch_size, 3, self.input_size, self.input_size)))
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {int(params)}")
