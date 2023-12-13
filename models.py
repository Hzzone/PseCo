import torch
import torch.nn as nn
import torchvision.ops as vision_ops
import torch.nn.functional as F
from ops.foundation_models.segment_anything.modeling.mask_decoder import MLP
from ops.foundation_models.segment_anything.modeling.common import LayerNorm2d
from ops.foundation_models.segment_anything.modeling.transformer import TwoWayTransformer

class ROIHeadMLP(nn.Module):
    def __init__(self):
        super(ROIHeadMLP, self).__init__()
        self.image_region_size = 7
        self.linear = nn.Sequential(nn.Linear(256 * self.image_region_size * self.image_region_size, 4096), nn.ReLU(True), nn.Linear(4096, 512))
        # self.linear = nn.Linear(256, 512)

    def forward(self, features, bboxes, prompts):
        image_embeddings = vision_ops.roi_align(features, [b.reshape(-1, 4) for b in bboxes],
                                        output_size=(self.image_region_size, self.image_region_size),
                                        spatial_scale=1 / 16, aligned=True)
        embeddings = self.linear(image_embeddings.flatten(1))
        # embeddings = self.linear(image_embeddings.mean(dim=(2, 3)).flatten(1))
        embeddings = embeddings.reshape(-1, bboxes[0].size(1), 512)
        embeddings = torch.cat([embeddings[i].unsqueeze(0).repeat(x.size(0), 1, 1) for i, x in enumerate(prompts)])
        prompts = torch.cat(prompts)
        pred_logits = (embeddings * prompts.unsqueeze(1)).sum(dim=-1)
        return pred_logits



class PointDecoder(nn.Module):
    def __init__(self, sam) -> None:
        super().__init__()
        transformer_dim = 256
        activation = nn.GELU
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )
        self.mask_tokens = nn.Embedding(1, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.transformer.load_state_dict(sam.mask_decoder.transformer.state_dict())
        self.output_upscaling.load_state_dict(sam.mask_decoder.output_upscaling.state_dict())
        self.output_hypernetworks_mlp.load_state_dict(sam.mask_decoder.output_hypernetworks_mlps[0].state_dict())
        from ops.foundation_models.segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
        embed_dim = 256
        self.image_embedding_size = (64, 64)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.nms_kernel_size = 3
        self.point_threshold = 0.1
        self.max_points = 1000

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, image_embeddings, masks=None):
        output_tokens = self.mask_tokens.weight[0].unsqueeze(0)
        sparse_embeddings = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        image_pe = self.get_dense_pe()
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, sparse_embeddings)
        src = src.transpose(1, 2).view(b, c, h, w)
        mask_tokens_out = hs[:, 0, :]
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlp(mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_embedding.shape
        pred_heatmaps = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        if self.training:
            return {'pred_heatmaps': pred_heatmaps}

        if masks is not None:
            pred_heatmaps *= masks

        with torch.no_grad():
            from ops.ops import _nms
            # pred_heatmaps_nms = _nms(pred_heatmaps.detach().sigmoid().clone(), self.nms_kernel_size)
            pred_heatmaps_nms = _nms(pred_heatmaps.detach().clone(), self.nms_kernel_size)
            pred_points, pred_points_score = torch.zeros(b, self.max_points, 2).cuda(), torch.zeros(b,
                                                                                                    self.max_points).cuda()
            m = 0
            for i in range(b):
                points = torch.nonzero((pred_heatmaps_nms[i] > self.point_threshold).squeeze())
                points = torch.flip(points, dims=(-1,))
                pred_points_score_ = pred_heatmaps_nms[i, 0, points[:, 1], points[:, 0]].flatten(0)

                idx = torch.argsort(pred_points_score_, dim=0, descending=True)[
                      :min(self.max_points, pred_points_score_.size(0))]
                # print(points.size(), pred_points_score_.size(),  idx, idx.max())
                points = points[idx]
                pred_points_score_ = pred_points_score_[idx]
                # print(points.size(), pred_points_score_.size(), pred_points_score_)
                # print(pred_points.size(), pred_points_score.size())
                # print(i)
                #
                pred_points[i, :points.size(0)] = points
                pred_points_score[i, :points.size(0)] = pred_points_score_
                m = max(m, points.size(0))
            # pred_points = (pred_points + 0.5) * 4
            pred_points = pred_points[:, :m]
            pred_points_score = pred_points_score[:, :m]
            pred_points = pred_points * 4

        return {'pred_heatmaps': pred_heatmaps,
                'pred_points': pred_points,
                'pred_points_score': pred_points_score,
                'pred_heatmaps_nms': pred_heatmaps_nms}
