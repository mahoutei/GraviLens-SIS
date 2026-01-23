class PhysicsPreprocessing(nn.Module):
    """
    CORRECTED: Compute full curvature information from Hessian

    Pipeline:
    1. Log transform: log(I + ε) to compress dynamic range
    2. Hessian computation: Get all second derivatives
    3. Curvature features: Laplacian ∇²f and Gaussian curvature det(H)
    4. Normalization & activation

    Output: 2-channel feature map [Laplacian, Determinant]
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

        # Separate normalization for each curvature channel
        self.norm_laplacian = nn.InstanceNorm2d(1, affine=True)
        self.norm_determinant = nn.InstanceNorm2d(1, affine=True)

        # Register second derivative kernels as buffers (non-trainable)
        self._register_derivative_kernels()

    def _register_derivative_kernels(self):
        """Create and register second derivative kernels"""
        # ∂²/∂x² kernel
        kernel_xx = torch.tensor([[0, 0, 0],
                                  [1, -2, 1],
                                  [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        # ∂²/∂y² kernel
        kernel_yy = torch.tensor([[0, 1, 0],
                                  [0, -2, 0],
                                  [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        # ∂²/∂x∂y kernel (mixed partial)
        kernel_xy = torch.tensor([[1, 0, -1],
                                  [0, 0, 0],
                                  [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) * 0.25

        self.register_buffer('kernel_xx', kernel_xx)
        self.register_buffer('kernel_yy', kernel_yy)
        self.register_buffer('kernel_xy', kernel_xy)

    def compute_hessian_features(self, image):
        """
        Compute full Hessian matrix and derive curvature features

        Returns:
            laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² (trace of Hessian)
            determinant: det(H) = f_xx·f_yy - f_xy² (Gaussian curvature)
        """
        # Ensure correct shape (B, 1, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(1)

        # Compute all second derivatives
        f_xx = F.conv2d(image, self.kernel_xx, padding=1)  # ∂²f/∂x²
        f_yy = F.conv2d(image, self.kernel_yy, padding=1)  # ∂²f/∂y²
        f_xy = F.conv2d(image, self.kernel_xy, padding=1)  # ∂²f/∂x∂y

        # Laplacian (trace of Hessian): detects blob-like structures
        laplacian = f_xx + f_yy

        # Determinant (Gaussian curvature): detects corners and edges
        # det(H) = f_xx·f_yy - f_xy²
        determinant = f_xx * f_yy - f_xy * f_xy

        return laplacian, determinant

    def forward(self, image):
        """
        Full preprocessing pipeline

        Args:
            image: (B, H, W) or (B, 1, H, W) input image

        Returns:
            features: (B, 2, H, W) with [Laplacian, Determinant] channels
        """
        # Step 1: Log transform to compress dynamic range
        log_image = torch.log(image + self.eps)

        # Step 2: Compute Hessian-based curvature features
        laplacian, determinant = self.compute_hessian_features(log_image)

        # Step 3: Normalize each feature independently
        laplacian_norm = self.norm_laplacian(laplacian)
        determinant_norm = self.norm_determinant(determinant)

        # Step 4: Activation (Tanh → Abs to get positive features)
        laplacian_feat = torch.abs(torch.tanh(laplacian_norm))
        determinant_feat = torch.abs(torch.tanh(determinant_norm))

        # Step 5: Stack into 2-channel feature map
        features = torch.cat([laplacian_feat, determinant_feat], dim=1)

        return features  # (B, 2, H, W)