from .backbones.vovnet import VoVNet
from .backbones.vovnetcp import VoVNetCP
from .necks.cp_fpn import CPFPN

__all__ = [
  "VoVNet",
  "VoVNetCP",
  "CPFPN"
]