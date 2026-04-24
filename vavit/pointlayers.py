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


class EmbeddingLayer(nn.Module):
    def __init__(self, cin, chidden, cout):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(cin, chidden, bias=False),
            nn.BatchNorm1d(chidden),
            nn.ReLU(inplace=True),
        )

        self.neigh_mlp = nn.Sequential(
            nn.Linear(cin, chidden, bias=False),
            nn.BatchNorm1d(chidden),
            nn.ReLU(inplace=True),
        )

        self.neigh_gate = nn.Sequential(
            nn.Linear(chidden, chidden, bias=True),
            nn.Sigmoid(),
        )

        self.final = nn.Linear(2 * chidden, cout, bias=False)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.ones_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, neighbors_emb):
        # Direct point embedding
        point_emb = self.point_mlp(x)

        # MLP on relative neighborhood features
        N, K, C = *neighbors_emb.shape, point_emb.shape[-1]
        x_neighbors = x[neighbors_emb]
        rel_coord = x_neighbors - x.unsqueeze(1)
        rel_coord = rel_coord.view(N * K, x.shape[-1])
        neigh_emb = self.neigh_mlp(rel_coord)

        # Gate on neighborhood
        gate = self.neigh_gate(neigh_emb)
        neigh_emb = neigh_emb * gate

        # Neighborhood embedding
        neigh_emb = neigh_emb.view(N, K, C)
        neigh_emb = neigh_emb.max(dim=1)[0]  # Max Pool over neighbors

        # Merge point and neighborhood embedding
        merge_emb = torch.cat((point_emb, neigh_emb), dim=1)

        # Final point embedding
        global_emb = self.final(merge_emb)

        return global_emb


class MergeHead(nn.Module):
    def __init__(self, cin, chidden, cout):
        super().__init__()

        self.bev_proj = nn.Sequential(
            nn.Linear(cin, chidden, bias=False),
            nn.BatchNorm1d(chidden),
            nn.ReLU(inplace=True),
        )

        self.point_proj = nn.Sequential(
            nn.Linear(cin, chidden, bias=False),
            nn.BatchNorm1d(chidden),
            nn.ReLU(inplace=True),
        )

        self.merge = nn.Sequential(
            nn.Linear(2 * chidden, cout, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * chidden, 2 * chidden, bias=True),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.ones_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, point_emb, bev_feat):
        point_emb = self.point_proj(point_emb)
        bev_feat = self.bev_proj(bev_feat)
        merge_feat = torch.cat((point_emb, bev_feat), dim=1)
        gate = self.gate(merge_feat)
        merge_feat = merge_feat * gate
        global_feat = self.merge(merge_feat)
        return global_feat
