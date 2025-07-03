import torch

from projects.StreamPETR.stream_petr.models.utils.misc import inverse_sigmoid, topk_gather, transform_reference_points
from projects.StreamPETR.stream_petr.models.utils.positional_encoding import (
    pos2posemb3d,
)

# Wrapper Classes for onnx conversion


class TrtPositionEmbeddingContainer(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(self, img_metas_pad, img_feats, intrinsics, img2lidar):
        mod = self.mod
        data = {
            "img_feats": img_feats,
            "intrinsics": intrinsics,
            "img2lidar": img2lidar,
        }
        location = mod.prepare_location(img_metas_pad, **data)
        return mod.pts_bbox_head.position_embeding(data, location, None, img_metas_pad)


class TrtEncoderContainer(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def forward(self, img):
        mod = self.mod
        return mod.extract_img_feat(img, 1).squeeze(1)


class TrtPtsHeadContainer(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = mod

    def _post_update_memory(
        self, data_ego_pose, data_timestamp, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec
    ):
        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        rec_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        head = self.mod.pts_bbox_head
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, head.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        head.memory_embedding = torch.cat([rec_memory, head.memory_embedding], dim=1)
        head.memory_timestamp = torch.cat([rec_timestamp, head.memory_timestamp], dim=1)
        head.memory_egopose = torch.cat([rec_ego_pose, head.memory_egopose], dim=1)
        head.memory_reference_point = torch.cat([rec_reference_points, head.memory_reference_point], dim=1)
        head.memory_velo = torch.cat([rec_velo, head.memory_velo], dim=1)
        head.memory_reference_point = transform_reference_points(
            head.memory_reference_point, data_ego_pose, reverse=False
        )
        head.memory_egopose = data_ego_pose.unsqueeze(1) @ head.memory_egopose

        # cast to float64 out-of-tensorrt
        # head.memory_timestamp -= data_timestamp.unsqueeze(-1).unsqueeze(-1)

    def _pts_head(self, x, pos_embed, cone, topk_indexes=None):
        head = self.mod.pts_bbox_head

        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)
        # don't do on-the-fly position_embedding
        # head.position_embeding(data, memory_center, topk_indexes, img_metas)

        memory = head.memory_embed(memory)

        # spatial_alignment in focal petr
        memory = head.spatial_alignment(memory, cone)
        pos_embed = head.featurized_pe(pos_embed, memory)

        reference_points = head.reference_points.weight
        reference_points, attn_mask, mask_dict = head.prepare_for_dn(B, reference_points, {}, None)
        query_pos = head.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        query_pos_in = query_pos.detach()
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = head.temporal_alignment(
            query_pos, tgt, reference_points
        )

        # transformer here is a little different from PETR
        outs_dec, _ = head.transformer(memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)
        outputs_classes = []
        outputs_coords = []
        reference = inverse_sigmoid(reference_points.clone())
        for lvl in range(outs_dec.shape[0]):
            outputs_class = head.cls_branches[lvl](outs_dec[lvl])
            tmp = head.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] * (head.pc_range[3:6] - head.pc_range[0:3]) + head.pc_range[0:3]
        )

        return (
            pos_embed,
            reference_points,
            tgt,
            temp_memory,
            temp_pos,
            query_pos,
            query_pos_in,
            outs_dec,
            all_cls_scores,
            all_bbox_preds,
            rec_ego_pose,
        )

    def forward(
        self,
        x,
        pos_embed,
        cone,
        data_timestamp,
        data_ego_pose,
        data_ego_pose_inv,
        memory_embedding,
        memory_reference_point,
        memory_timestamp,
        memory_egopose,
        memory_velo,
    ):
        # x[1, 6, 256, 16, 44]
        # pos_embed[1, 4224, 256]
        # cone[1, 4224, 8]
        head = self.mod.pts_bbox_head

        # memory update before head
        # memory_timestamp += data_timestamp.unsqueeze(-1).unsqueeze(-1)
        memory_egopose = data_ego_pose_inv.unsqueeze(1) @ memory_egopose
        memory_reference_point = transform_reference_points(memory_reference_point, data_ego_pose_inv, reverse=False)

        head.memory_timestamp = memory_timestamp[:, : head.memory_len]
        head.memory_reference_point = memory_reference_point[:, : head.memory_len]
        head.memory_embedding = memory_embedding[:, : head.memory_len]
        head.memory_egopose = memory_egopose[:, : head.memory_len]
        head.memory_velo = memory_velo[:, : head.memory_len]

        (
            pos_embed,
            reference_points,
            tgt,
            temp_memory,
            temp_pos,
            query_pos,
            query_pos_in,
            outs_dec,
            all_cls_scores,
            all_bbox_preds,
            rec_ego_pose,
        ) = self._pts_head(x, pos_embed, cone, topk_indexes=None)

        # memory update after head
        self._post_update_memory(data_ego_pose, data_timestamp, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec)

        return (
            all_cls_scores.flatten(0, 2)
            .unsqueeze(0)
            .transpose(2, 1),  # This is to make the tensors easier to handle in cpp
            all_bbox_preds.flatten(0, 2)
            .unsqueeze(0)
            .transpose(2, 1),  # This is to make the tensors easier to handle in cpp
            head.memory_embedding,
            head.memory_reference_point,
            head.memory_timestamp,
            head.memory_egopose,
            head.memory_velo,
            reference_points,
            tgt,
            temp_memory,
            temp_pos,
            query_pos,
            query_pos_in,
            outs_dec,
        )
