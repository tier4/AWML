from functools import partial

import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from addict import Dict
from libs.pointrope import PointROPE
from models.builder import MODELS
from models.modules import MLP, Embedding, PointModule, PointSequential
from models.scatter.functional import argsort, segment_csr, unique
from models.utils.structure import Point
from timm.layers import DropPath


class PointROPEAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        rope_freq,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        # pointrope
        self.rope = PointROPE(freq=rope_freq)

    def forward(self, point: Point):
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = point.get_padding_and_inverse(self.patch_size)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]  # [N, C]

        ## apply pointrope
        pos = point.grid_coord[order]  # [N, 3]
        pos = pos.reshape(-1, 3).unsqueeze(0)

        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, H, C // H).transpose(0, 1)[None]  # [1, H, N, head_dim]
        k = k.reshape(-1, H, C // H).transpose(0, 1)[None]  # [1, H, N, head_dim]

        # workround to make pointrope cuda float32 happy
        q = self.rope(q.float(), pos).to(q.dtype)  # [1, H, N, head_dim]
        k = self.rope(k.float(), pos).to(k.dtype)  # [1, H, N, head_dim]

        # assemble input for flash attention
        qkv_rotated = torch.stack(
            [
                q.squeeze(0).transpose(0, 1),
                k.squeeze(0).transpose(0, 1),
                v.reshape(-1, H, C // H),
            ],
            dim=1,
        )  # [N, 3, H, head_dim]

        if torch.onnx.is_in_onnx_export():
            assert (qkv_rotated.shape[0] % K) == 0
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = qkv_rotated.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            # attn
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            attn = F.softmax(attn, dim=-1)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_rotated,
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)

        feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_conv=True,
        enable_attn=True,
        rope_freq=100.0,
        export_mode=False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.export_mode = export_mode

        self.enable_conv = enable_conv
        self.enable_attn = enable_attn

        if self.enable_conv:
            if export_mode:
                from SparseConvolution.sparse_conv import SubMConv3d
            else:
                from spconv.pytorch import SubMConv3d

            self.conv = PointSequential(
                SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(
                norm_layer(channels),
            )

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = PointROPEAttention(
                channels=channels,
                patch_size=patch_size,
                rope_freq=rope_freq,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        re_serialization=False,
        serialization_order="z",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

        self.re_serialization = re_serialization
        self.serialization_order = serialization_order

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError("[gird_coord] or [coord, grid_size] should be include in the Point")
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")

        # Pack batch id and coordinates into a single 1D key for clustering.
        # NOTE(original): grid_coord = torch.bitwise_or(grid_coord, point.batch.view(-1, 1) << 48)
        grid_coord_i = grid_coord.to(torch.int64)
        gx = grid_coord_i[:, 0]
        gy = grid_coord_i[:, 1]
        gz = grid_coord_i[:, 2]
        b = point.batch.to(torch.int64)
        packed = gx + gy * (1 << 16) + gz * (1 << 32) + b * (1 << 48)

        if torch.onnx.is_in_onnx_export():
            unique_keys, cluster, counts, _ = unique(packed)
            # Sort points by cluster id (inverse_indices) to make them contiguous per cluster.
            indices = argsort(cluster)
        else:
            unique_keys, cluster, counts = torch.unique(
                packed,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            # indices of point sorted by cluster, for torch_scatter.segment_csr
            _, indices = torch.sort(cluster)

        # Unpack to (M, 3) grid coords (drop batch component)
        # NOTE(original): grid_coord = torch.bitwise_and(grid_coord, ((1 << 48) - 1))
        key_wo_batch = torch.remainder(unique_keys, (1 << 48))
        gx = torch.remainder(key_wo_batch, (1 << 16))
        gy = torch.remainder(torch.div(key_wo_batch, (1 << 16), rounding_mode="trunc"), (1 << 16))
        gz = torch.remainder(torch.div(key_wo_batch, (1 << 32), rounding_mode="trunc"), (1 << 16))
        grid_coord = torch.stack([gx, gy, gz], dim=1).to(grid_coord_i.dtype)

        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]

        if not torch.onnx.is_in_onnx_export():
            scatter_feat = torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce)
            scatter_coord = torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean")
        else:
            scatter_feat = segment_csr(self.proj(point.feat)[indices], idx_ptr, self.reduce)
            scatter_coord = segment_csr(point.coord[indices], idx_ptr, "mean")

        point_dict = Dict(
            feat=scatter_feat,
            coord=scatter_coord,
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )

        if "origin_coord" in point.keys():
            if not torch.onnx.is_in_onnx_export():
                point_dict["origin_coord"] = torch_scatter.segment_csr(
                    point.origin_coord[indices], idx_ptr, reduce="mean"
                )
            else:
                point_dict["origin_coord"] = segment_csr(point.origin_coord[indices], idx_ptr, "mean")
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            if not torch.onnx.is_in_onnx_export():
                point_dict["color"] = torch_scatter.segment_csr(point.color[indices], idx_ptr, reduce="mean")
            else:
                point_dict["color"] = segment_csr(point.color[indices], idx_ptr, "mean")
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        if "mask" in point.keys():
            if not torch.onnx.is_in_onnx_export():
                point_dict["mask"] = (
                    torch_scatter.segment_csr(point.mask[indices].float(), idx_ptr, reduce="mean") > 0.5
                )
            else:
                point_dict["mask"] = segment_csr(point.mask[indices].float(), idx_ptr, "mean") > 0.5

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)

        if self.re_serialization:
            point.serialization(order=self.serialization_order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


@MODELS.register_module("LitePT")
class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        export_mode=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.export_mode = export_mode

        self.enc_conv = enc_conv
        self.enc_attn = enc_attn
        self.dec_conv = dec_conv
        self.dec_attn = dec_attn

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm

        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            export_mode=self.export_mode,
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                        export_mode=self.export_mode,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s],
                            export_mode=self.export_mode,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """
        data_dict is the batched input point cloud, it should contain as least:
        1. feat [N, input_dim]: input feature for the point cloud
        2. grid_coord [N, 3]: voxelized coordinate after grid sampling
           or/and
           coord [N, 3]: original coordinate + grid_size: grid_size used for grid sampling
        3. offset [batch_size]: separator of point clouds in batched data
           or/and
           batch [N]: batch index of each point
        """
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point

    def export_forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point["serialized_depth"] = data_dict["serialized_depth"]
            point["serialized_code"] = data_dict["serialized_code"]
            point["serialized_order"] = data_dict["serialized_order"]
            point["serialized_inverse"] = data_dict["serialized_inverse"]
            point["sparse_shape"] = data_dict["sparse_shape"]
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point
