# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Tuple, Union

import torch
import torch.nn as nn

from ..components.flatten import Flatten
from ..components.lsa import FeedForwardBlockLSA, TransformerLSABlock
from ..components.spt_lensformer import ShiftPatchTokenizer
from ..physics.lensformer_physics import PhysicsInformedEncoder

class Lensformer(nn.Module):
    """
    Lensformer: Relativistic Physics-Informed Vision Transformer.
    Reference: Velôso et al., NeurIPS ML4PS Workshop 2023.
    """
    def __init__(self, image_size: int, patch_size: int, embed_dim: int,
                 in_channels: int, num_classes: int, num_heads: int,
                 num_hidden_neurons: int, num_hidden_layers: int,
                 transformer_activation, feedforward_activation,
                 num_transformer_blocks: int,
                 device: Union[str, torch.device],
                 dropout_rate: float = 0.1):
        super().__init__()
        self.image_size = image_size
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)

        self.initial_tokenizer   = ShiftPatchTokenizer(
            image_size, embed_dim, patch_size, in_channels,
            "diagonal", device)
        self.secondary_tokenizer = ShiftPatchTokenizer(
            image_size, embed_dim, patch_size, in_channels,
            "diagonal", device)
        self.tertiary_tokenizer  = ShiftPatchTokenizer(
            image_size, embed_dim, patch_size, in_channels,
            "diagonal", device)

        num_patches = self.initial_tokenizer.get_num_patches()

        self.encoder = PhysicsInformedEncoder(
            image_size=image_size, patch_size=patch_size,
            embedding_dim=embed_dim, num_patches=num_patches,
            num_heads=num_heads, hidden_dim=num_hidden_neurons,
            transformer_block_activation_function=feedforward_activation,
            num_transformer_block=num_transformer_blocks,
            device=device, k_max=1.2, k_min=0.8)

        self.transformer_blocks = nn.ModuleList([
            TransformerLSABlock(
                embedding_dim=embed_dim, num_head=num_heads,
                num_patches=num_patches,
                num_hidden_neurons=num_hidden_neurons,
                num_hidden_layers=1,
                activation_function=transformer_activation,
                device=device, dropout=dropout_rate)
            for _ in range(num_transformer_blocks)
        ])

        flat_dim = (num_patches + 1) * embed_dim
        self.flatten_layer = Flatten(flat_dim)
        self.feedforward_layer = FeedForwardBlockLSA(
            in_dim=flat_dim, out_dim=num_classes,
            hidden_dim=num_hidden_neurons,
            num_hidden_layers=num_hidden_layers,
            activation_function=feedforward_activation,
            task_type='multiclass', dropout=dropout_rate)

    def forward(self, images: torch.Tensor,
                distortions: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
        images      = images.to(self.device)
        distortions = distortions.to(self.device)
        B = images.size(0)

        dis_patches     = self.tertiary_tokenizer(
            distortions.reshape(B, 1, self.image_size, self.image_size))
        initial_patches = self.initial_tokenizer(
            images.reshape(B, 1, self.image_size, self.image_size))

        info, lens_corrected = self.encoder(images, initial_patches,
                                            distortions)
        source = lens_corrected.view(B, 1, self.image_size, self.image_size)
        source_patches = self.secondary_tokenizer(source)

        concatenated = source_patches
        for block in self.transformer_blocks:
            concatenated = block(key=dis_patches,
                                 value=initial_patches,
                                 query=source_patches)

        logits = self.feedforward_layer(self.flatten_layer(concatenated))
        return info, source, logits
