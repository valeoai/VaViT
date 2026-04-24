# Copyright 2026 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from .pointlayers import EmbeddingLayer, MergeHead
from .transformer import Transformer


class Segmenter(nn.Module):
    def __init__(
        self,
        emb_cin,
        emb_chidden,
        emb_num_layers,
        vit_dim,
        vit_depth,
        vit_num_heads,
        vit_expansion,
        merge_chidden,
        merge_cout,
        nb_class,
        drop_path=0.0,
    ):
        super().__init__()

        # Pre-normalization
        self.norm = nn.BatchNorm1d(emb_cin)

        # Embedding layer
        self.embed = self.get_embed(
            emb_num_layers,
            emb_cin,
            emb_chidden,
            vit_dim,
        )

        # Transformer in BEV
        self.vit = Transformer(
            vit_dim,
            vit_depth,
            vit_num_heads,
            vit_expansion,
            drop_path,
        )

        # Padding token
        self.pad_token = nn.Parameter(torch.zeros((vit_dim)), requires_grad=True)

        # Point head to merge BEV and point features
        self.merge = MergeHead(vit_dim, merge_chidden, merge_cout)

        # Classif.
        self.classif = nn.Linear(merge_cout, nb_class, bias=True)

        #
        self.init_weights()

    def get_embed(self, emb_num_layers, emb_cin, emb_chidden, vit_dim):
        embed = nn.ModuleList()

        if emb_num_layers > 1:
            channels = [(emb_cin, emb_chidden, emb_chidden)]
            for i in range(1, emb_num_layers - 1):
                channels += [(emb_chidden, emb_chidden, emb_chidden)]
            channels += [(emb_chidden, emb_chidden, vit_dim)]
        else:
            channels = [(emb_cin, emb_chidden, vit_dim)]

        for i, (cin, chidden, cout) in enumerate(channels):
            embed.append(EmbeddingLayer(cin, chidden, cout))
        return embed

    def init_weights(self):
        for m in [self.norm, self.classif]:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, feats, neighbors, xy_bev, idx_to_bev, idx_occ, idx_pad, bev_splits
    ):

        # Pre-normalization
        point_emb = self.norm(feats)

        # Point embedding
        for embed in self.embed:
            point_emb = embed(point_emb, neighbors)

        # Dimension: batch_size x channels x sequence length
        B, C, M = len(bev_splits), point_emb.shape[1], bev_splits[0]

        # Init BEV map with zeros
        bev_feat = torch.zeros(
            (B * M, C), dtype=point_emb.dtype, device=point_emb.device
        )

        # Actual projection
        idx_bev_expand = idx_to_bev.unsqueeze(1).expand(-1, C)
        bev_feat.scatter_reduce_(
            0, idx_bev_expand, point_emb, reduce="amax", include_self=False
        )

        # Padding token
        if self.training:
            bev_feat = bev_feat.clone()
        if idx_pad is not None:
            pad_token = (
                self.pad_token.to(point_emb.dtype).unsqueeze(0).expand(len(idx_pad), -1)
            )
            bev_feat.index_add_(0, idx_pad, pad_token)

        # Positional encoding
        rope_coords = torch.zeros(
            (B * M, 2), dtype=point_emb.dtype, device=point_emb.device
        )
        rope_coords[idx_occ] = xy_bev.to(point_emb.dtype)
        rope_coords = rope_coords.reshape(B, M, 2)

        # Transformer in BEV
        bev_feat = bev_feat.reshape(B, M, -1)
        bev_feat = self.vit(bev_feat, rope_coords)

        # Lift bev feat and merge with point feat
        point_emb = self.merge(point_emb, bev_feat.reshape(B * M, C)[idx_to_bev])

        # Classif
        return self.classif(point_emb)
