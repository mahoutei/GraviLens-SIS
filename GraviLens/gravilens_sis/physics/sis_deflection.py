class SimpleSISDeflection:
    """SIS deflection using potential gradient (Lensformer approach)"""
    def __init__(self, eps=1e-8, core_radius=0.01):
        self.eps = eps
        self.core_radius = core_radius

    def calculate_deflection(self, x, y, k):
        """
        Calculate SIS deflection via potential gradient

        Ψ = κ · √(x² + y² + ε²)
        α = ∇Ψ = κ · ∇√(x² + y² + ε²)
        """
        # Expand k if needed
        if k.ndim == 1:
            k = k.unsqueeze(-1).expand(-1, x.shape[-1])

        # Compute potential components
        r = torch.sqrt(x**2 + y**2 + self.core_radius**2)

        # Analytical gradient: ∂Ψ/∂x = κ · x/r
        alpha_x = k * x / r
        alpha_y = k * y / r

        return alpha_x, alpha_y