# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import math

import torch
import torch.nn as nn

from ..components.feedforward import FeedForward
from ..components.flatten import Flatten
from ..components.htransformer import HierarchicalTransformerStage
from ..components.spt import ShiftedPatchTokenization
from ..physics.sis_encoder import SISPhysicsInformedEncoder

class GraviLens(nn.Module):
    """
    GraviLens with SIS physics (formerly GraviLensSIS).

    Pipeline:
      1. ShiftedPatchTokenization (initial patches from lensed image)
      2. SISPhysicsInformedEncoder -> source-plane reconstruction
      3. ShiftedPatchTokenization (secondary patches from source image)
      4. HierarchicalTransformerStage with cross-attention
         (query=initial, key/value=source patches)
      5. FeedForward classifier head
    """
    def __init__(self, image_size: int, patch_size: int, embed_dim: int,
                 in_channels: int, num_classes: int, num_heads: int,
                 num_hidden_neurons: int, num_hidden_layers: int,
                 transformer_activation, feedforward_activation,
                 num_transformer_blocks: int, device: torch.device,
                 dropout_rate: float = 0.1, window_size: int = 4):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.device     = device

        self.initial_tokenizer   = ShiftedPatchTokenization(
            image_size, patch_size, embed_dim, in_channels, device)
        self.secondary_tokenizer = ShiftedPatchTokenization(
            image_size, patch_size, embed_dim, in_channels, device)

        num_patches = self.initial_tokenizer.get_num_patches()
        self.H = self.W = int(math.sqrt(num_patches))

        if self.H % window_size != 0:
            orig_ws = window_size
            for ws in [2, 4, 8]:
                if self.H % ws == 0:
                    window_size = ws
                    break
            print(f"  Adjusted window_size: {orig_ws} -> {window_size} "
                  f"for H={self.H}")
        self.window_size = window_size

        print(f"  GraviLens config:")
        print(f"    Image {image_size}x{image_size}, "
              f"Patches {self.H}x{self.W} ({num_patches} total)")
        print(f"    embed_dim={embed_dim}, heads={num_heads}, "
              f"blocks={num_transformer_blocks}, window={window_size}")

        self.encoder = SISPhysicsInformedEncoder(
            image_size, patch_size, embed_dim, num_patches, num_heads,
            num_hidden_neurons, transformer_activation,
            num_transformer_blocks, device, window_size=window_size)

        self.transformer_stage = HierarchicalTransformerStage(
            dim=embed_dim, depth=num_transformer_blocks,
            num_heads=num_heads, window_size=window_size,
            mlp_ratio=4., dropout=dropout_rate)

        self.flatten_layer = Flatten((num_patches + 1) * embed_dim)
        self.feedforward_layer = FeedForward(
            self.flatten_layer.num_neurons_flatten, num_classes,
            feedforward_activation, num_hidden_neurons,
            num_hidden_layers, dropout=dropout_rate)

    def invert_lens(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        patches = self.initial_tokenizer(
            images.reshape(B, 1, self.image_size, self.image_size))
        return self.encoder(images, patches)

    def predict_lens_parameters(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        patches = self.initial_tokenizer(
            images.reshape(B, 1, self.image_size, self.image_size))
        out = self.encoder.transformer(patches, self.H, self.W)
        return self.encoder.sis_predictor(out)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        init_patches = self.initial_tokenizer(
            images.reshape(B, 1, self.image_size, self.image_size))
        source_img = self.encoder(images, init_patches)
        src_patches = self.secondary_tokenizer(
            source_img.reshape(B, 1, self.image_size, self.image_size))
        out = self.transformer_stage(
            x=init_patches, H=self.H, W=self.W,
            key=src_patches, value=src_patches)
        return self.feedforward_layer(self.flatten_layer(out))
