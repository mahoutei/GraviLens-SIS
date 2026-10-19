# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).

class GradientAccumulator:
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self):
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False

    def reset(self):
        self.current_step = 0
