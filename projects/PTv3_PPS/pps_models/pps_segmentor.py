"""
PPSSegmentor: PTv3 backbone + PPS prototype-based head.

Supported experimental features (controlled via head config):
  - supervised_class_ids: partial supervision — only listed classes are supervised
  - adaptive_ema: per-class EMA momentum inversely scaled by class frequency
      → rare classes update prototypes faster (more responsive to few examples)
  - rare_temperature / rare_class_ids: two-temperature system
      → rare classes use a smaller temperature (sharper decision boundary)

Two-phase fine-tuning from an existing DefaultSegmentorV2 checkpoint:
  Phase 1 (freeze_backbone=True, ~10 epochs): warm up the PPS head only.
  Phase 2 (freeze_backbone=False, ~40 epochs): full fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import MODELS, build_model
from models.utils.structure import Point


# ---------------------------------------------------------------------------
# PartProtoHead
# ---------------------------------------------------------------------------

class PartProtoHead(nn.Module):
    """
    Args:
        feat_dim: backbone output channels (64 for PTv3)
        num_classes: total number of semantic classes
        ignore_index: GT label value excluded from all losses
        temperature: base cosine similarity temperature (lower = sharper)
        ema_momentum: prototype EMA decay; higher = slower update (0.99 default)
        conf_threshold: min softmax confidence to include a point in EMA update
        proto_loss_weight: weight for supervised prototype contrastive loss
        af3_loss_weight: weight for adaptive focal foreground loss
        ortho_loss_weight: weight for prototype orthogonality regularization
        supervised_class_ids: if set, remap all other GT labels to ignore_index
            (partial supervision). None = full supervision.
        adaptive_ema: if True, per-class EMA momentum scales with class frequency:
            momentum_c = ema_momentum + (1 - ema_momentum) * (freq_c / max_freq)
            Rare classes (low freq) → lower momentum → larger updates per step.
        rare_class_ids: list of class IDs considered "rare" for two-temperature.
        rare_temperature: temperature used for rare_class_ids columns in logits.
            If None, all classes use the same `temperature`.
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
        adaptive_ema: bool = False,
        rare_class_ids: list | None = None,
        rare_temperature: float | None = None,
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
        self.adaptive_ema = adaptive_ema
        self.rare_class_ids = set(rare_class_ids) if rare_class_ids is not None else set()
        self.rare_temperature = rare_temperature

        # Build per-class inverse-temperature vector [C] for two-temperature system
        inv_temps = torch.full((num_classes,), 1.0 / temperature)
        if rare_temperature is not None and rare_class_ids:
            for cid in rare_class_ids:
                inv_temps[cid] = 1.0 / rare_temperature
        self.register_buffer("inv_temps", inv_temps)

        # Running class frequency estimate for adaptive EMA (updated each forward)
        self.register_buffer("class_freq", torch.ones(num_classes) / num_classes)

        # Learnable class prototypes [C, D]
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _cosine_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """[N, D] → [N, C] cosine similarity logits (per-class temperature)."""
        feat_n = F.normalize(feat, dim=1)           # [N, D]
        proto_n = F.normalize(self.prototypes, dim=1)  # [C, D]
        sim = feat_n @ proto_n.T                    # [N, C]
        return sim * self.inv_temps.unsqueeze(0)    # broadcast [N, C] * [1, C]

    # ------------------------------------------------------------------
    # Prototype EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_prototypes(
        self,
        feat: torch.Tensor,
        segment: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        probs = logits.softmax(dim=1)
        conf, _ = probs.max(dim=1)

        # Update running class frequency (EMA over batches)
        counts = torch.bincount(
            segment[segment != self.ignore_index].clamp(min=0),
            minlength=self.num_classes,
        ).float()
        batch_freq = counts / counts.sum().clamp(min=1)
        self.class_freq = 0.99 * self.class_freq + 0.01 * batch_freq

        for cls_id in range(self.num_classes):
            mask = (segment == cls_id) & (conf >= self.conf_threshold)
            if mask.sum() == 0:
                continue
            cls_mean = F.normalize(feat[mask], dim=1).mean(0)

            # Per-class momentum: adaptive or fixed
            if self.adaptive_ema:
                max_freq = self.class_freq.max().clamp(min=1e-6)
                # Rare class (low freq) → momentum closer to ema_momentum (fast update)
                # Common class (high freq) → momentum closer to 1.0 (slow update)
                momentum = self.ema_momentum + (1.0 - self.ema_momentum) * (
                    self.class_freq[cls_id] / max_freq
                )
            else:
                momentum = self.ema_momentum

            if not self.initialized[cls_id]:
                self.prototypes.data[cls_id].copy_(cls_mean)
                self.initialized[cls_id] = True
            else:
                self.prototypes.data[cls_id] = (
                    momentum * self.prototypes.data[cls_id]
                    + (1.0 - momentum) * cls_mean
                )

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _ce_loss(self, logits: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, segment, ignore_index=self.ignore_index)

    def _proto_contrast_loss(
        self, feat: torch.Tensor, segment: torch.Tensor
    ) -> torch.Tensor:
        """Supervised contrastive: align each point's feature toward its true prototype."""
        feat_n = F.normalize(feat, dim=1)
        proto_n = F.normalize(self.prototypes, dim=1)
        # Use base temperature only for contrastive (not per-class)
        logits = feat_n @ proto_n.T / self.temperature
        return F.cross_entropy(logits, segment, ignore_index=self.ignore_index)

    def _af3_loss(self, logits: torch.Tensor, segment: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Focal Foreground loss.
        gamma_c = 3 * (1 - freq_c): rare classes → gamma≈3, common → gamma≈0.
        """
        valid = segment != self.ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0

        seg_v = segment[valid]
        logits_v = logits[valid]

        counts = torch.bincount(seg_v, minlength=self.num_classes).float()
        freq = counts / counts.sum().clamp(min=1)
        gamma = (3.0 * (1.0 - freq)).clamp(0.0, 3.0)

        log_probs = F.log_softmax(logits_v, dim=1)
        true_log_p = log_probs.gather(1, seg_v.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - true_log_p.exp()) ** gamma[seg_v]
        return (focal_w * (-true_log_p)).mean()

    def _ortho_loss(self) -> torch.Tensor:
        """||P @ P^T - I||_F^2 — prototype orthogonality regularization."""
        proto_n = F.normalize(self.prototypes, dim=1)
        gram = proto_n @ proto_n.T
        eye = torch.eye(self.num_classes, device=gram.device)
        return (gram - eye).pow(2).mean()

    # ------------------------------------------------------------------
    # Partial supervision
    # ------------------------------------------------------------------

    def _apply_partial_supervision(self, segment: torch.Tensor) -> torch.Tensor:
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
        logits = self._cosine_logits(feat)

        if segment is None:
            return {"seg_logits": logits}

        seg = self._apply_partial_supervision(segment)

        if self.training:
            self._update_prototypes(feat.detach(), seg, logits.detach())

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
        head: PartProtoHead config dict
        freeze_backbone: if True, backbone parameters are frozen (Phase 1)
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
        ckpt = torch.load(weight_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
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
