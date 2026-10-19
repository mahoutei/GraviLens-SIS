# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch

class Lookahead:
    """Lookahead optimizer wrapper (Zhang et al., 2019)."""
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state.setdefault(p, {})
                param_state['slow_param'] = torch.zeros_like(p.data)
                param_state['slow_param'].copy_(p.data)
        self.step_counter = 0

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        if self.step_counter >= self.k:
            self.step_counter = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    slow = self.state[p]['slow_param']
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'state': self.state,
                'step_counter': self.step_counter}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.state = state_dict['state']
        self.step_counter = state_dict['step_counter']
