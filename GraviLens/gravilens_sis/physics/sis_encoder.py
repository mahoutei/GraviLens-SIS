class SISPhysicsInformedEncoder(nn.Module):
    """
    COMPLETELY FIXED Physics-Informed Encoder

    Architecture Flow (CORRECTED):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Raw Image (B, H, W)                                          │
    │    ↓                                                             │
    │ 2. Physics Preprocessing                                        │
    │    - Log transform                                              │
    │    - Hessian computation → [Laplacian, Determinant]             │
    │    - Normalization & activation                                 │
    │    ↓                                                             │
    │ 3. Physics Features (B, 2, H, W)  ← TWO CHANNELS!               │
    │    ↓                                                             │
    │ 4. Tokenization (via external tokenizer)                        │
    │    - Shifted patch tokenization on physics features             │
    │    - NOT on raw images!                                         │
    │    ↓                                                             │
    │ 5. Physics Patches (B, N+1, C)                                  │
    │    ↓                                                             │
    │ 6. Transformer Processing                                       │
    │    - Processes physics-informed patches                         │
    │    - Extracts high-level physics features                       │
    │    ↓                                                             │
    │ 7. κ-Predictor                                                  │
    │    - Predicts convergence map from physics features             │
    │    ↓                                                             │
    │ 8. SIS Deflection                                               │
    │    - α(θ) = κ·(θ/|θ|)                                          │
    │    ↓                                                             │
    │ 9. Source Reconstruction                                        │
    │    - β = θ - α(θ)                                               │
    │    - Inverse splatting                                          │
    │    ↓                                                             │
    │ 10. Output: Delensed Image (B, H, W)                            │
    └─────────────────────────────────────────────────────────────────┘

    KEY FIX: Physics features now actually flow through the pipeline!
    """

    def __init__(self, image_size, patch_size, embed_dim, num_patches,
                 num_heads, num_hidden_neurons, activation_function,
                 num_transformer_blocks, device, window_size=4,
                 use_physics_preprocessing=True):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.device = device
        self.use_physics_preprocessing = use_physics_preprocessing

        # Physical constants
        self.pixel_scale = 0.101  # arcsec per pixel
        self.profile_size = image_size
        self.H = self.W = int(math.sqrt(num_patches))

        # Validate and adjust window size for compatibility
        if self.H % window_size != 0:
            original_ws = window_size
            for ws in [2, 4, 8]:
                if self.H % ws == 0:
                    window_size = ws
                    break
            print(f"⚠️  Adjusted window_size from {original_ws} to {window_size} for H={self.H}")

        self.window_size = window_size

        # ========== FIXED: Physics Preprocessing Module ==========
        self.physics_preprocessing = PhysicsPreprocessing(eps=1e-8)

        print(f"\n✅ FixedSISPhysicsInformedEncoder Configuration:")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Patch grid: {self.H}x{self.W} ({num_patches} patches)")
        print(f"   Window size: {window_size}x{window_size}")
        print(f"   Embed dim: {embed_dim}")
        print(f"   Physics preprocessing: {'ENABLED' if use_physics_preprocessing else 'DISABLED'}")
        print(f"   Output: 2-channel physics features [Laplacian, Determinant]")

        # Create coordinate grids for deflection calculation
        half_size = (self.profile_size - 1) / 2
        coords_1d = torch.linspace(-half_size, half_size, self.profile_size,
                                   device=device) * self.pixel_scale
        grid_x, grid_y = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        self.flat_grid_x = grid_x.flatten()
        self.flat_grid_y = grid_y.flatten()

        # Import required components from your codebase
        # (You'll need to adjust these imports based on your actual module structure)
        from __main__ import (
            HierarchicalTransformerStage,
            SimpleSISDeflection,
            HybridSISPredictor
        )

        # Transformer for extracting physics features
        self.transformer = HierarchicalTransformerStage(
            dim=embed_dim,
            depth=num_transformer_blocks,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.,
            dropout=0.1
        )

        # SIS physics components
        self.sis_deflection = SimpleSISDeflection(eps=1e-6, core_radius=0.01)
        self.sis_predictor = HybridSISPredictor(
            embed_dim,
            image_size=image_size,
            device=device
        )

        print(f"✅ Encoder initialized successfully!\n")

    def forward(self, input_images: torch.Tensor, physics_patches: torch.Tensor) -> torch.Tensor:
        """
        CORRECTED Forward Pass - Physics features ACTUALLY used!

        Args:
            input_images: (B, H, W) raw lensed galaxy images
            physics_patches: (B, N+1, C) patch embeddings from PHYSICS FEATURES
                            ^^^^^^^^^^^^^ CRITICAL: These come from preprocessed features!

        Returns:
            output_images: (B, H, W) reconstructed source plane images

        Pipeline:
        1. [Already done externally] Preprocess → Physics features (2 channels)
        2. [Already done externally] Tokenize physics features → Physics patches
        3. [This function] Process physics patches → High-level features
        4. [This function] Predict κ-map → Deflection → Source reconstruction
        """
        batch_size = input_images.shape[0]

        # ========== VERIFICATION: Check that physics patches are used ==========
        # This is just a sanity check - in production you can remove this
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"DEBUG: input_images shape: {input_images.shape}")
            print(f"DEBUG: physics_patches shape: {physics_patches.shape}")
            print(f"DEBUG: physics_patches should come from 2-channel features!")

        # ========== STEP 1: Process Physics Patches with Transformer ==========
        # This is where the MAIN FIX is: we use physics_patches, not raw patches!
        transformer_output = self.transformer(physics_patches, self.H, self.W)

        # ========== STEP 2: Predict κ-map from Physics-Informed Features ==========
        k_map = self.sis_predictor(transformer_output)
        k_flat = k_map[:, 0, :, :].view(batch_size, -1)

        # Validate κ-map
        if torch.isnan(k_map).any():
            print("⚠️ WARNING: NaN detected in κ-map! Clamping to safe values.")
            k_map = torch.nan_to_num(k_map, nan=1.0)
            k_flat = k_map[:, 0, :, :].view(batch_size, -1)

        # ========== STEP 3: Calculate SIS Deflection Field ==========
        # α(θ) = κ · (θ / |θ|)
        alpha_x, alpha_y = self.sis_deflection.calculate_deflection(
            self.flat_grid_x.unsqueeze(0).expand(batch_size, -1),
            self.flat_grid_y.unsqueeze(0).expand(batch_size, -1),
            k_flat
        )

        # ========== STEP 4: Source Plane Reconstruction ==========
        # Lensing equation: β = θ - α(θ)
        source_x = self.flat_grid_x.unsqueeze(0) - alpha_x
        source_y = self.flat_grid_y.unsqueeze(0) - alpha_y

        # Convert to pixel coordinates
        source_x_pixels = (source_x / self.pixel_scale) + (self.profile_size - 1) / 2
        source_y_pixels = (source_y / self.pixel_scale) + (self.profile_size - 1) / 2

        # Get flux from input (lensed) image
        input_flat = input_images.view(batch_size, -1)

        # Initialize output source images
        output_images = torch.zeros(batch_size, self.profile_size, self.profile_size,
                                    device=self.device)

        # ========== STEP 5: Inverse Splatting (Bilinear) ==========
        for b in range(batch_size):
            # Get valid source positions (within image bounds)
            valid_mask = (
                (source_x_pixels[b] >= 0) & (source_x_pixels[b] < self.profile_size) &
                (source_y_pixels[b] >= 0) & (source_y_pixels[b] < self.profile_size)
            )

            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # Get source positions and fluxes for valid pixels
            src_x = source_x_pixels[b, valid_indices]
            src_y = source_y_pixels[b, valid_indices]
            flux_values = input_flat[b, valid_indices]

            # Bilinear splatting (distribute flux to 4 nearest pixels)
            x0 = torch.floor(src_x).long()
            y0 = torch.floor(src_y).long()
            x1 = x0 + 1
            y1 = y0 + 1

            # Bilinear weights
            wx1 = src_x - x0.float()
            wy1 = src_y - y0.float()
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            # Clamp to valid range
            x0 = torch.clamp(x0, 0, self.profile_size - 1)
            y0 = torch.clamp(y0, 0, self.profile_size - 1)
            x1 = torch.clamp(x1, 0, self.profile_size - 1)
            y1 = torch.clamp(y1, 0, self.profile_size - 1)

            # Splat flux to 4 neighboring pixels with bilinear weights
            output_images[b].index_put_(
                (y0, x0),
                flux_values * wx0 * wy0,
                accumulate=True
            )
            output_images[b].index_put_(
                (y0, x1),
                flux_values * wx1 * wy0,
                accumulate=True
            )
            output_images[b].index_put_(
                (y1, x0),
                flux_values * wx0 * wy1,
                accumulate=True
            )
            output_images[b].index_put_(
                (y1, x1),
                flux_values * wx1 * wy1,
                accumulate=True
            )

        # ========== STEP 6: Normalize ==========
        max_values = torch.amax(output_images, dim=(1, 2), keepdim=True)
        max_values = torch.where(max_values == 0, torch.ones_like(max_values) * 1e-8, max_values)
        output_images = output_images / max_values

        return output_images

    def get_predicted_parameters(self, input_images: torch.Tensor, physics_patches: torch.Tensor) -> torch.Tensor:
        """
        Get predicted κ-map from physics-preprocessed images

        Args:
            input_images: (B, H, W) lensed images (used for reference only)
            physics_patches: (B, N+1, C) patch embeddings from physics features

        Returns:
            k_map: (B, 1, H, W) predicted convergence map
        """
        # Process physics patches through transformer
        transformer_output = self.transformer(physics_patches, self.H, self.W)

        # Predict κ-map
        k_map = self.sis_predictor(transformer_output)

        return k_map

    def get_physics_features(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        Extract raw physics-preprocessed features for analysis

        Args:
            input_images: (B, H, W) lensed images

        Returns:
            physics_features: (B, 2, H, W) preprocessed features
                             [:, 0, :, :] = Laplacian
                             [:, 1, :, :] = Determinant
        """
        if input_images.dim() == 3:
            input_images = input_images.unsqueeze(1)

        physics_features = self.physics_preprocessing(input_images)

        return physics_features

    def enable_debug_mode(self):
        """Enable debug mode for verification"""
        self._debug_mode = True
        print("✅ Debug mode enabled")

    def disable_debug_mode(self):
        """Disable debug mode"""
        self._debug_mode = False
        print("✅ Debug mode disabled")