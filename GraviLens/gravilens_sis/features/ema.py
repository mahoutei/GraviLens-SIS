class ModelEMA:
    """Exponential Moving Average of model weights"""
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)