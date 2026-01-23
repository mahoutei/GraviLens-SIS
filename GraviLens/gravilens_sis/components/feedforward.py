class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, input_dim, output_dim, activation, hidden_dim, num_layers, task_type='multi_classification', dropout=0.1):
        super().__init__()
        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)