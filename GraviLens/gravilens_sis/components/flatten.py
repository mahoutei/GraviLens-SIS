class Flatten(nn.Module):
    """Flatten layer"""
    def __init__(self, num_neurons_flatten):
        super().__init__()
        self.num_neurons_flatten = num_neurons_flatten

    def forward(self, x):
        return x.view(x.size(0), -1)