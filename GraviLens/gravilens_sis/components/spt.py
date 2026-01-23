# %% ============================================================================
# SECTION 5: PATCH TOKENIZATION - FIXED
# ============================================================================

class ShiftedPatchTokenization(nn.Module):
    """
    CORRECTED: Tokenize PREPROCESSED features instead of raw images

    This now handles 2-channel input from physics preprocessing
    """
    def __init__(self, image_size, patch_size, embed_dim, in_channels=2, device=None):
        """
        Args:
            in_channels: 2 for physics features [Laplacian, Determinant]
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.embed_dim = embed_dim
        self.image_width, self.image_height = image_size
        self.patch_width, self.patch_height = patch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.image_width % self.patch_width != 0 or self.image_height % self.patch_height != 0:
            raise ValueError("Image dimensions must be divisible by patch dimensions")

        self.num_patches = int((self.image_width // self.patch_width) *
                              (self.image_height // self.patch_height))

        # Shift parameters
        self.delta_x = self.patch_width // 2
        self.delta_y = self.patch_height // 2
        self.num_transformations = 4

        # Total channels: in_channels * (original + 4 shifts)
        self.total_channels = in_channels * (self.num_transformations + 1)

        # Projection layer
        self.projection = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_width, self.patch_height),
            stride=(self.patch_width, self.patch_height)
        ).to(device)

        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.embed_dim)))
        self.positional_encoding = nn.Parameter(
            torch.zeros((1, 1 + self.num_patches, self.embed_dim))
        )

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def translate_image(self, image, delta_x, delta_y):
        """Apply spatial shift augmentation"""
        import torchvision.transforms.functional as TF
        translated = TF.affine(image, angle=0, translate=(delta_x, delta_y),
                              fill=0, scale=1, shear=0)
        return translated.to(self.device)

    def forward(self, features):
        """
        Tokenize physics features (NOT raw images!)

        Args:
            features: (B, 2, H, W) physics-preprocessed features

        Returns:
            patches: (B, N+1, embed_dim) patch embeddings with CLS token
        """
        batch_size = features.size(0)

        # Apply shifted patch tokenization on PHYSICS FEATURES
        shift_left_up = self.translate_image(features, -self.delta_x, -self.delta_y)
        shift_right_up = self.translate_image(features, self.delta_x, -self.delta_y)
        shift_left_down = self.translate_image(features, -self.delta_x, self.delta_y)
        shift_right_down = self.translate_image(features, self.delta_x, self.delta_y)

        # Concatenate: (B, 2*5, H, W)
        concatenated = torch.cat(
            (features, shift_left_up, shift_right_up, shift_left_down, shift_right_down),
            dim=1
        )

        # Project to patches
        projected_patches = self.projection(concatenated)  # (B, embed_dim, H_p, W_p)
        patches = projected_patches.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Normalize
        patches = self.layer_norm(patches)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls, patches), dim=1)

        # Add positional encoding
        patches = patches + self.positional_encoding

        return patches

    def get_num_patches(self):
        return self.num_patches