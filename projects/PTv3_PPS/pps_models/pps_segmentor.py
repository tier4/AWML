"""
PPSSegmentor: PTv3 backbone + PPS prototype-based head.

Replaces DefaultSegmentorV2's linear seg_head with PartProtoHead, which adds:
  - Cosine-similarity classification via learnable class prototypes
  - EMA prototype updates during training (high-confidence points only)
  - Proto contrast loss: supervised contrastive alignment of features to prototypes
  - AF³ loss: adaptive focal foreground loss (rare classes get higher focal weight)
  - Ortho loss: prototype orthogonality regularization (prevents collapse)

Two-phase fine-tuning from an existing DefaultSegmentorV2 checkpoint:
  Phase 1 (freeze_backbone=True, ~10 epochs): warm up the PPS head only.
  Phase 2 (freeze_backbone=False, ~40 epochs): full fine-tuning.

Prototype warm-start: seg_head.weight [num_classes, feat_dim] from the loaded
checkpoint initializes prototypes directly — no random restart.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# PTv3 project must be on sys.path (handled by PTv3_PPS/train.py)
from models.builder import MODELS, build_model
from models.utils.structure import Point


# ---------------------------------------------------------------------------
# PartProtoHead
# ---------------------------------------------------------------------------

class PartProtoHead(nn.Module):
    """
    Prototype-based segmentation head implementing PPS losses.

    Args:
        feat_dim: backbone output channels (64 for PTv3)
        num_classes: total number of semantic classes
        ignore_index: GT label value excluded from all losses
        temperature: cosine similarity temperature; lower = sharper predictions
        ema_momentum: prototype EMA decay (0.99 = slow update)
        conf_threshold: min softmax confidence for a point to contribute to EMA
        proto_loss_weight: weight for proto contrast loss
        af3_loss_weight: weight for AF³ adaptive focal loss
        ortho_loss_weight: weight for prototype orthogonality loss
        supervised_class_ids: optional list of class IDs to supervise; all other
            labeled points are remapped to ignore_index (partial supervision mode).
            None = full supervision (default).
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        ignore_index: int = -1,
        temperature: float = 0.07,
        ema_momentum: float = 0.99,
        conf_threshold: float = 0.6,
        proto_loss_weight: float = 0.2,
        af3_loss_weight: float = 1.0,
        ortho_loss_weight: float = 0.1,
        supervised_class_ids: list | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.conf_threshold = conf_threshold
        self.proto_loss_weight = proto_loss_weight
        self.af3_loss_weight = af3_loss_weight
        self.ortho_loss_weight = ortho_loss_weight
        self.supervised_class_ids = (
            set(supervised_class_ids) if supervised_class_ids is not None else None
        )

        # Learnable class prototypes [C, D] — warm-started from seg_head.weight
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        # Track which classes have been seen at least once (guards EMA init)
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _cosine_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """[N, D] → [N, C] cosine similarity logits scaled by 1/temperature."""
        feat_n = F.normalize(feat, dim=1)
        proto_n = F.normalize(self.prototypes, dim=1)
        return feat_n @ proto_n.T / self.temperature

    # ------------------------------------------------------------------
    # Prototype EMA update (training only, no gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_prototypes(
        self,
        feat: torch.Tensor,
        segment: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Update prototypes via EMA using high-confidence GT-labeled points."""
        probs = logits.softmax(dim=1)
        conf, _ = probs.max(dim=1)  # [N]

        for cls_id in range(self.num_classes):
            mask = (segment == cls_id) & (conf >= self.conf_threshold)
            if mask.sum() == 0:
                continue
            cls_mean = F.normalize(feat[mask], dim=1).mean(0)  # [D]
            if not self.initialized[cls_id]:
                self.prototypes.data[cls_id].copy_(cls_mean)
                self.initialized[cls_id] = True
            else:
                self.prototypes.data[cls_id] = (
                    self.ema_momentum * self.prototypes.data[cls_id]
                    + (1.0 - self.ema_momentum) * cls_mean
                )

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _ce_loss(self, logits: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, segment, ignore_index=self.ignore_index)

    def _proto_contrast_loss(
        self, feat: torch.Tensor, segment: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised contrastive loss: align each point's feature toward its
        true-class prototype using cosine logits.
        """
        feat_n = F.normalize(feat, dim=1)
        proto_n = F.normalize(self.prototypes, dim=1)
        logits = feat_n @ proto_n.T / self.temperature  # [N, C]
        return F.cross_entropy(logits, segment, ignore_index=self.ignore_index)

    def _af3_loss(self, logits: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Focal Foreground (AF³) loss.

        Per-class frequency in the current batch determines gamma:
            gamma_c = 3 * (1 - freq_c)
        Rare classes (low freq) → high gamma → stronger focus on hard examples.
        Common classes (high freq) → low gamma → near-standard CE.
        """
        valid = segment != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0

        seg_v = segment[valid]
        logits_v = logits[valid]

        # Per-class frequency in this batch
        counts = torch.bincount(seg_v, minlength=self.num_classes).float()
        freq = counts / counts.sum().clamp(min=1)  # [C]
        gamma = (3.0 * (1.0 - freq)).clamp(0.0, 3.0)  # [C]

        log_probs = F.log_softmax(logits_v, dim=1)  # [N_valid, C]
        true_log_p = log_probs.gather(1, seg_v.unsqueeze(1)).squeeze(1)  # [N_valid]
        true_p = true_log_p.exp()
        focal_w = (1.0 - true_p) ** gamma[seg_v]  # [N_valid]
        return (focal_w * (-true_log_p)).mean()

    def _ortho_loss(self) -> torch.Tensor:
        """
        Orthogonality regularization: ||P @ P^T - I||_F^2 / C^2.
        Encourages prototype diversity and prevents collapse of rare-class prototypes
        into the feature space of common classes.
        """
        proto_n = F.normalize(self.prototypes, dim=1)  # [C, D]
        gram = proto_n @ proto_n.T  # [C, C]
        eye = torch.eye(self.num_classes, device=gram.device)
        return (gram - eye).pow(2).mean()

    # ------------------------------------------------------------------
    # Partial supervision helper
    # ------------------------------------------------------------------

    def _apply_partial_supervision(self, segment: torch.Tensor) -> torch.Tensor:
        """Remap non-supervised class labels to ignore_index."""
        if self.supervised_class_ids is None:
            return segment
        supervised = torch.zeros_like(segment, dtype=torch.bool)
        for cls_id in self.supervised_class_ids:
            supervised |= segment == cls_id
        out = segment.clone()
        out[~supervised & (segment != self.ignore_index)] = self.ignore_index
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feat: torch.Tensor,
        segment: torch.Tensor | None = None,
    ) -> dict:
        logits = self._cosine_logits(feat)  # [N, C]

        if segment is None:
            # Test mode: no labels available
            return {"seg_logits": logits}

        seg = self._apply_partial_supervision(segment)

        if self.training:
            self._update_prototypes(feat.detach(), seg, logits.detach())

        # Compute losses
        ce_loss = self._ce_loss(logits, seg)
        proto_loss = self._proto_contrast_loss(feat, seg) * self.proto_loss_weight
        af3_loss = self._af3_loss(logits, seg) * self.af3_loss_weight
        ortho_loss = self._ortho_loss() * self.ortho_loss_weight
        total = ce_loss + proto_loss + af3_loss + ortho_loss

        if self.training:
            return {"loss": total}
        return {"loss": total, "seg_logits": logits}


# ---------------------------------------------------------------------------
# PPSSegmentor
# ---------------------------------------------------------------------------

@MODELS.register_module()
class PPSSegmentor(nn.Module):
    """
    PTv3 backbone + PartProtoHead.

    Args:
        num_classes: number of semantic classes
        backbone_out_channels: PTv3 decoder output channels (64)
        backbone: backbone config dict (same as DefaultSegmentorV2)
        head: PartProtoHead config dict (all PartProtoHead __init__ kwargs)
        freeze_backbone: if True, backbone parameters are frozen (Phase 1 training)
        weight_path: path to a DefaultSegmentorV2 checkpoint for warm-start
    """

    def __init__(
        self,
        num_classes: int,
        backbone_out_channels: int,
        backbone: dict,
        head: dict,
        freeze_backbone: bool = False,
        weight_path: str | None = None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.head = PartProtoHead(
            feat_dim=backbone_out_channels,
            num_classes=num_classes,
            **head,
        )
        if weight_path is not None:
            self._load_pretrained(weight_path)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _load_pretrained(self, weight_path: str) -> None:
        """
        Load backbone weights and warm-start prototypes from a DefaultSegmentorV2
        checkpoint (model_best.pth).

        The checkpoint may be a raw state_dict or wrapped under 'state_dict'.
        Backbone keys are expected as 'backbone.*'; prototype seed from 'seg_head.weight'.
        """
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # Strip 'module.' prefix from DDP checkpoints if present
        state = {k.removeprefix("module."): v for k, v in state.items()}

        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"[PPSSegmentor] backbone: {len(missing)} missing keys")
        if unexpected:
            print(f"[PPSSegmentor] backbone: {len(unexpected)} unexpected keys")

        # Warm-start prototypes from the existing linear head weights [C, D]
        if "seg_head.weight" in state:
            with torch.no_grad():
                self.head.prototypes.data.copy_(state["seg_head.weight"])
            print("[PPSSegmentor] Prototypes warm-started from seg_head.weight")
        else:
            print("[PPSSegmentor] WARNING: seg_head.weight not found; prototypes randomly initialized")

    def forward(self, input_dict: dict) -> dict:
        point = Point(input_dict)
        point = self.backbone(point)
        feat = point.feat if isinstance(point, Point) else point
        segment = input_dict.get("segment")
        return self.head(feat, segment)
