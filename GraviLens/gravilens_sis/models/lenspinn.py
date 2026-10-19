# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Tuple, Union

import timm
import torch
import torch.nn as nn

from ..components.lsa import FeedForwardBlockLSA
from ..components.spt_lensformer import ShiftPatchTokenizer
from ..physics.lensformer_physics import ChangedPhysicsInformedEncoder

class LensPINN(nn.Module):
    """
    LensPINN: Physics-Informed Neural Network for dark matter morphology.
    Two variants controlled by ``mode``:
      - 'small': MobileNetV3-Small backbone
      - 'large': EfficientNet-B0 backbone
    """
    def __init__(self, image_size: int, patch_size: int, embed_dim: int,
                 in_channels: int, num_classes: int, num_heads: int,
                 num_hidden_neurons: int, num_hidden_layers: int,
                 transformer_activation, feedforward_activation,
                 num_transformer_blocks: int,
                 device: Union[str, torch.device],
                 mode: str = 'small',
                 dropout_rate: float = 0.1):
        super().__init__()
        if mode not in ('small', 'large'):
            raise ValueError("mode must be 'small' or 'large'.")

        self.image_size = image_size
        self.mode = mode
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)

        self.initial_tokenizer = ShiftPatchTokenizer(
            image_size, embed_dim, patch_size, in_channels,
            "diagonal", device)

        num_patches = self.initial_tokenizer.get_num_patches()

        self.encoder = ChangedPhysicsInformedEncoder(
            image_size=image_size, patch_size=patch_size,
            embedding_dim=embed_dim, num_patches=num_patches,
            num_heads=num_heads, hidden_dim=num_hidden_neurons,
            transformer_block_activation_function=feedforward_activation,
            num_transformer_block=num_transformer_blocks,
            device=device, k_max=1.2, k_min=0.8)

        backbone_name = ("efficientnet_b0" if mode == 'large'
                         else "mobilenetv3_small_050")
        self.decoder1 = timm.create_model(backbone_name, pretrained=True,
                                          num_classes=0, in_chans=1)
        self.decoder2 = timm.create_model(backbone_name, pretrained=True,
                                          num_classes=0, in_chans=1)

        self._upsample = nn.Upsample(size=224, mode='bilinear',
                                     align_corners=False)

        with torch.no_grad():
            _dummy = torch.zeros(1, 1, 224, 224)
            backbone_out = self.decoder1(_dummy).shape[1]

        self.fc = FeedForwardBlockLSA(
            in_dim=backbone_out * 2, out_dim=num_classes,
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

        distortions = distortions.view(B, 1, self.image_size, self.image_size)
        initial_patches = self.initial_tokenizer(
            images.reshape(B, 1, self.image_size, self.image_size))

        theta_E, lens_corrected = self.encoder(images, initial_patches,
                                               distortions)
        source = lens_corrected.view(B, 1, self.image_size, self.image_size)

        imgs_up   = self._upsample(images.view(B, 1, self.image_size, self.image_size))
        source_up = self._upsample(source)
        dis_up    = self._upsample(distortions)

        image_feats  = self.decoder1(imgs_up)
        source_feats = self.decoder1(source_up)
        dis_feats    = self.decoder2(dis_up)

        combined = torch.cat([image_feats - source_feats, dis_feats], dim=1)
        logits = self.fc(combined)
        return theta_E, source, logits
