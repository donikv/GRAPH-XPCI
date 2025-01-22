from typing import Optional
from transformers import ViTConfig, ViTForImageClassification
from torch import nn
import torch
import torchvision
from common.vit_mae import *
from common.embeddings import interpolate_pos_embed
from timm.models.layers import trunc_normal_


def add_vit_args(parser):
    add_vitmae_args(parser)

@DeprecationWarning
def build_model_old(args, pretrained_model: Optional[ViTMAEForPreTraining]=None):
    size = args.size if args.crop_size is None else args.crop_size

    config = ViTConfig(
        image_size=size,
        num_channels=args.in_channels,
        patch_size=args.patch_size,
        hidden_size=args.latent_dim,
        num_hidden_layers=args.encoder_depth,
        num_attention_heads=args.encoder_num_heads,
        intermediate_size=args.encoder_intermediate_size,
        norm_pix_loss=args.patch_norm,
        decoder_hidden_size=args.decoder_latent_dim,
        decoder_num_hidden_layers=args.decoder_depth,
        decoder_num_attention_heads=args.decoder_num_heads,
        decoder_intermediate_size=args.decoder_intermediate_size,
        num_labels=args.num_classes
    )

    model = ViTForImageClassification(config)
    if pretrained_model is not None:
        state_dict = pretrained_model.vit.state_dict()
        model.vit.load_state_dict(state_dict=state_dict)
    return model


def build_model(args, global_pool=False, pretrained_model: Optional[MaskedAutoencoderViT]=None):
    size = args.size if args.crop_size is None else args.crop_size

    model = VisionTransformer(global_pool=global_pool,
                              img_size=size,
                              patch_size=args.patch_size,
                              embed_dim=args.latent_dim,
                              depth=args.encoder_depth,
                              in_chans=args.in_channels,
                              num_heads=args.encoder_num_heads,
                              mlp_ratio=int(args.encoder_intermediate_size / args.latent_dim),
                              qkv_bias=True,
                              num_classes=args.num_classes,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    model = load_weights_from_mae(model, pretrained_model)
    return model

def load_weights_from_mae(model, checkpoint_model):
        state_dict = model.state_dict()
        checkpoint_model = checkpoint_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        return model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome