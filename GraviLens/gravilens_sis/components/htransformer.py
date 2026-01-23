class HierarchicalTransformerStage(nn.Module):
    """Hierarchical Transformer with window attention - FIXED"""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, H, W, key=None, value=None):
        """
        Args:
            x: (B, N+1, C) where N = H*W (includes CLS token)
            H, W: spatial dimensions
        """
        # Validate dimensions
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"Spatial dimensions H={H}, W={W} must be divisible by window_size={self.window_size}. "
                f"Current patch grid is {H}x{W}, which gives {H % self.window_size} and {W % self.window_size} remainders."
            )

        for blk in self.blocks:
            x = blk(x, H, W, key, value)
        return x