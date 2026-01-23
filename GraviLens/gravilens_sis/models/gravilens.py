class GraviLensSIS(nn.Module):
    """
    FIXED GraviLens model with complete interface

    Now includes:
    - forward() for classification
    - invert_lens() for source reconstruction
    - predict_lens_parameters() for κ-map prediction
    """
    def __init__(self, image_size, patch_size, embed_dim, in_channels,
                 num_classes, num_heads, num_hidden_neurons, num_hidden_layers,
                 transformer_activation, feedforward_activation,
                 num_transformer_blocks, device, dropout_rate=0.1,
                 window_size=4, use_physics_preprocessing=True):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.use_physics_preprocessing = use_physics_preprocessing

        # ========== Physics Preprocessing Pipeline ==========
        # Stage 1: Physics preprocessing (Log → Hessian → 2 channels)
        self.physics_preprocessing = PhysicsPreprocessing(eps=1e-8)

        # Stage 2: Tokenize physics features (2-channel input)
        self.physics_tokenizer = ShiftedPatchTokenization(
            image_size, patch_size, embed_dim, in_channels=2, device=device
        )

        # For secondary path (delensed image tokenization)
        self.secondary_tokenizer = ShiftedPatchTokenization(
            image_size, patch_size, embed_dim, in_channels=1, device=device
        )

        # IMPORTANT: Store as initial_tokenizer for compatibility
        self.initial_tokenizer = self.physics_tokenizer

        num_patches = self.physics_tokenizer.get_num_patches()
        self.H = self.W = int(math.sqrt(num_patches))

        # Encoder with physics pipeline
        self.encoder = SISPhysicsInformedEncoder(
            image_size, patch_size, embed_dim, num_patches, num_heads,
            num_hidden_neurons, transformer_activation, num_transformer_blocks,
            device, window_size=window_size,
            use_physics_preprocessing=use_physics_preprocessing
        )

        self.transformer_stage = HierarchicalTransformerStage(
            dim=embed_dim, depth=num_transformer_blocks, num_heads=num_heads,
            window_size=window_size, mlp_ratio=4., dropout=dropout_rate
        )

        self.flatten_layer = Flatten((num_patches + 1) * embed_dim)
        self.feedforward_layer = FeedForward(
            self.flatten_layer.num_neurons_flatten, num_classes,
            feedforward_activation, num_hidden_neurons, num_hidden_layers,
            task_type='multi_classification', dropout=dropout_rate
        )

    def forward(self, images):
        """
        Main forward pass for classification

        Pipeline:
        1. Preprocess image → Physics features (2 channels)
        2. Tokenize physics features → Physics patches
        3. Encoder processes physics patches → Delensed image
        4. Tokenize delensed image → Secondary patches
        5. Cross-attention (query=physics, key/value=delensed)
        6. Classification

        Args:
            images: (B, H, W) input galaxy images

        Returns:
            predictions: (B, num_classes) class logits
        """
        batch_size = images.size(0)

        # Step 1: Physics preprocessing
        physics_features = self.physics_preprocessing(
            images.reshape(batch_size, 1, self.image_size, self.image_size)
        )  # (B, 2, 64, 64)

        # Step 2: Tokenize physics features
        physics_patches = self.physics_tokenizer(physics_features)

        # Step 3: Encoder uses physics patches to get delensed images
        lens_corrected_images = self.encoder(images, physics_patches)

        # Step 4: Tokenize delensed images
        lens_corrected_patches = self.secondary_tokenizer(
            lens_corrected_images.reshape(batch_size, 1, self.image_size, self.image_size)
        )

        # Step 5: Cross-attention
        output_patches = self.transformer_stage(
            x=physics_patches,  # Query: physics-informed features
            H=self.H, W=self.W,
            key=lens_corrected_patches,    # Key: delensed features
            value=lens_corrected_patches   # Value: delensed features
        )

        # Step 6: Classification
        flattened_patches = self.flatten_layer(output_patches)
        final_predictions = self.feedforward_layer(flattened_patches)

        return final_predictions

    # ========================================================================
    # NEW METHOD 1: GRAVITATIONAL LENS INVERSION
    # ========================================================================
    def invert_lens(self, images):
        """
        Reconstruct source galaxy from lensed observation

        This applies the physics-based gravitational lens inversion to
        reconstruct the unlensed source galaxy from the observed lensed image.

        Physics: β = θ - α(θ)
        Where:
        - θ = image plane position (observed)
        - α = deflection angle (from κ-map)
        - β = source plane position (reconstructed)

        Args:
            images: (B, H, W) or (B, 1, H, W) lensed galaxy images

        Returns:
            delensed_images: (B, H, W) reconstructed source plane images

        Example:
            >>> lensed_galaxy = torch.randn(1, 64, 64).to(device)
            >>> source = model.invert_lens(lensed_galaxy)
            >>> print(source.shape)  # (1, 64, 64)
        """
        batch_size = images.size(0)

        # Ensure correct input shape (B, 1, H, W)
        if images.dim() == 3:
            images = images.reshape(batch_size, 1, self.image_size, self.image_size)
        elif images.dim() == 4 and images.size(1) != 1:
            images = images.reshape(batch_size, 1, self.image_size, self.image_size)

        # Step 1: Preprocess to get physics features
        physics_features = self.physics_preprocessing(images)

        # Step 2: Tokenize physics features
        physics_patches = self.physics_tokenizer(physics_features)

        # Step 3: Get delensed reconstruction from encoder
        # The encoder applies: β = θ - α(θ) using SIS deflection
        delensed_images = self.encoder(images.squeeze(1), physics_patches)

        return delensed_images

    # ========================================================================
    # NEW METHOD 2: CONVERGENCE MAP PREDICTION
    # ========================================================================
    def predict_lens_parameters(self, images):
        """
        Predict convergence (κ) map from lensed images

        The κ-map represents the projected mass distribution of the lens,
        which determines the gravitational deflection strength.

        For SIS (Singular Isothermal Sphere):
        - κ(r) is approximately constant
        - Deflection: α = κ · (θ/|θ|)
        - Higher κ → stronger lensing → larger Einstein ring

        Args:
            images: (B, H, W) or (B, 1, H, W) lensed galaxy images

        Returns:
            kappa_map: (B, 1, H, W) predicted convergence map

        Example:
            >>> lensed_galaxy = torch.randn(1, 64, 64).to(device)
            >>> kappa = model.predict_lens_parameters(lensed_galaxy)
            >>> print(kappa.shape)  # (1, 1, 64, 64)
            >>> print(kappa.mean())  # Should be around 0.8-1.2 for physical lenses
        """
        batch_size = images.size(0)

        # Ensure correct input shape (B, 1, H, W)
        if images.dim() == 3:
            images = images.reshape(batch_size, 1, self.image_size, self.image_size)
        elif images.dim() == 4 and images.size(1) != 1:
            images = images.reshape(batch_size, 1, self.image_size, self.image_size)

        # Step 1: Preprocess to get physics features
        physics_features = self.physics_preprocessing(images)

        # Step 2: Tokenize physics features
        physics_patches = self.physics_tokenizer(physics_features)

        # Step 3: Get κ-map from encoder
        kappa_map = self.encoder.get_predicted_parameters(images.squeeze(1), physics_patches)

        return kappa_map