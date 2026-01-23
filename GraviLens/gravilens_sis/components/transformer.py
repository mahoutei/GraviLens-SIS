class TransformerBlock(nn.Module):
    """Transformer block with proper window attention and MLP - FIXED"""
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, H, W, key=None, value=None):
        """
        Args:
            x: (B, N+1, C) where N = H*W (includes CLS token)
            H, W: spatial dimensions after patch embedding
        """
        B, L, C = x.shape

        # Separate CLS token and patches
        cls_token = x[:, 0:1, :]  # (B, 1, C)
        x_patches = x[:, 1:, :]   # (B, H*W, C)

        shortcut = x_patches
        x_patches = self.norm1(x_patches)

        # Reshape to spatial dimensions
        x_patches = x_patches.view(B, H, W, C)

        # Partition into windows
        x_windows = self.window_partition(x_patches, self.window_size)  # (B*nW, window_size*window_size, C)

        # Apply window attention
        attn_windows = self.attn(x_windows)  # (B*nW, window_size*window_size, C)

        # Merge windows back
        x_patches = self.window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        x_patches = x_patches.view(B, H * W, C)

        # Residual connection
        x_patches = shortcut + x_patches

        # Add CLS token back
        x = torch.cat([cls_token, x_patches], dim=1)  # (B, N+1, C)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x, window_size):
        """
        Partition feature map into non-overlapping windows.
        Args:
            x: (B, H, W, C)
            window_size: window size
        Returns:
            windows: (B*num_windows, window_size*window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        Reverse of window partition.
        Args:
            windows: (B*num_windows, window_size*window_size, C)
            window_size: Window size
            H: Height of feature map
            W: Width of feature map
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x