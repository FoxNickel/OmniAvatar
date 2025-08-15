import torch
from typing import Tuple, Union
import torch
from einops import rearrange
from torch import nn
import deepspeed
checkpoint = deepspeed.checkpointing.checkpoint
from ..utils.args_config import args


def make_triple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    value = (value,) * 3 if isinstance(value, int) else value
    assert len(value) == 3
    return value

# 主要功能：
# 将输入的高维音频特征（如 [batch, channel, time, height, width]）按照指定 patch 大小分块（patchify）。
# 将每个 patch 展平成一维后，用全连接层（nn.Linear）投影到指定维度（dim）。
# 可选地对输出做 LayerNorm 归一化。
class AudioPack(nn.Module):
    def __init__(
            self,
            in_channels: int,
            patch_size: Union[int, Tuple[int, int, int]],
            dim: int,
            layernorm=False,
    ):
        super().__init__()
        t, h, w = make_triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)
        if layernorm:
            self.norm_out = nn.LayerNorm(dim)
        else:
            self.norm_out = None
        
        # print(f"[AudioPack] in_channels: {in_channels}, t, h, w: {t}, {h}, {w}")
        # print(f"[AudioPack] patch_size: {self.patch_size}")
        # print(f"[AudioPack] proj: {self.proj}")
        # if self.norm_out is not None:
        #     print(f"[AudioPack] norm_out: {self.norm_out}")

    def forward(self, vid: torch.Tensor):
        print(f"[AudioPack forward] use_checkpoint: {args.use_checkpoint}, training: {self.training}")
        if args.use_checkpoint and self.training:
            return checkpoint(self._forward, vid)
        else:
            return self._forward(vid)

    def _forward(self, vid: torch.Tensor,) -> torch.Tensor:
        t, h, w = self.patch_size
        print(f"[AudioPack forward] vid shape: {vid.shape}, t, h, w: {t}, {h}, {w}")
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        if self.norm_out is not None:
            vid = self.norm_out(vid)
        return vid